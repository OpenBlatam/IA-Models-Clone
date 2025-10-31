"""
Factory System
Factory patterns for creating modular components
"""

import logging
from typing import Dict, Any, List, Optional, Type, TypeVar, Callable, Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import inspect

from .interfaces import (
    IOptimizer, IModel, ITrainer, IInferencer, IMonitor, IBenchmarker,
    IConfigurable, ILoggable, IMeasurable
)
from .base import BaseModule

logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class FactoryConfig:
    """Configuration for factory"""
    name: str
    component_type: str
    implementation: str
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    priority: int = 0

class ComponentFactory(ABC, Generic[T]):
    """Abstract factory for components"""
    
    def __init__(self):
        self._creators: Dict[str, Type[T]] = {}
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._instances: Dict[str, T] = {}
    
    @abstractmethod
    def get_component_type(self) -> str:
        """Get the type of component this factory creates"""
        pass
    
    def register(self, name: str, component_class: Type[T], config: Optional[Dict[str, Any]] = None) -> None:
        """Register a component class"""
        self._creators[name] = component_class
        if config:
            self._configs[name] = config
        logger.info(f"Registered {self.get_component_type()} class: {name}")
    
    def create(self, name: str, instance_name: Optional[str] = None, **kwargs) -> Optional[T]:
        """Create a component instance"""
        if name not in self._creators:
            logger.error(f"Unknown {self.get_component_type()} class: {name}")
            return None
        
        component_class = self._creators[name]
        config = self._configs.get(name, {})
        config.update(kwargs)
        
        try:
            instance = component_class(instance_name or name, config)
            if instance_name:
                self._instances[instance_name] = instance
            logger.info(f"Created {self.get_component_type()} instance: {name}")
            return instance
        except Exception as e:
            logger.error(f"Failed to create {self.get_component_type()} {name}: {e}")
            return None
    
    def get_instance(self, name: str) -> Optional[T]:
        """Get existing instance"""
        return self._instances.get(name)
    
    def list_available(self) -> List[str]:
        """List available component classes"""
        return list(self._creators.keys())
    
    def list_instances(self) -> List[str]:
        """List created instances"""
        return list(self._instances.keys())
    
    def get_config(self, name: str) -> Dict[str, Any]:
        """Get configuration for a component class"""
        return self._configs.get(name, {})

class OptimizerFactory(ComponentFactory[IOptimizer]):
    """Factory for optimization components"""
    
    def get_component_type(self) -> str:
        return "optimizer"
    
    def create_optimizer(self, name: str, level: str, **kwargs) -> Optional[IOptimizer]:
        """Create optimizer with specific level"""
        config = {
            "level": level,
            **kwargs
        }
        return self.create(name, **config)
    
    def create_adaptive_optimizer(self, **kwargs) -> Optional[IOptimizer]:
        """Create adaptive optimizer"""
        return self.create("adaptive", **kwargs)
    
    def create_memory_optimizer(self, **kwargs) -> Optional[IOptimizer]:
        """Create memory optimizer"""
        return self.create("memory", **kwargs)
    
    def create_quantization_optimizer(self, **kwargs) -> Optional[IOptimizer]:
        """Create quantization optimizer"""
        return self.create("quantization", **kwargs)

class ModelFactory(ComponentFactory[IModel]):
    """Factory for model components"""
    
    def get_component_type(self) -> str:
        return "model"
    
    def create_transformer(self, **kwargs) -> Optional[IModel]:
        """Create transformer model"""
        return self.create("transformer", **kwargs)
    
    def create_cnn(self, **kwargs) -> Optional[IModel]:
        """Create CNN model"""
        return self.create("cnn", **kwargs)
    
    def create_rnn(self, **kwargs) -> Optional[IModel]:
        """Create RNN model"""
        return self.create("rnn", **kwargs)
    
    def create_hybrid(self, **kwargs) -> Optional[IModel]:
        """Create hybrid model"""
        return self.create("hybrid", **kwargs)
    
    def load_pretrained(self, path: str, **kwargs) -> Optional[IModel]:
        """Load pretrained model"""
        config = {"path": path, **kwargs}
        return self.create("pretrained", **config)

class TrainerFactory(ComponentFactory[ITrainer]):
    """Factory for training components"""
    
    def get_component_type(self) -> str:
        return "trainer"
    
    def create_standard_trainer(self, **kwargs) -> Optional[ITrainer]:
        """Create standard trainer"""
        return self.create("standard", **kwargs)
    
    def create_distributed_trainer(self, **kwargs) -> Optional[ITrainer]:
        """Create distributed trainer"""
        return self.create("distributed", **kwargs)
    
    def create_federated_trainer(self, **kwargs) -> Optional[ITrainer]:
        """Create federated trainer"""
        return self.create("federated", **kwargs)
    
    def create_reinforcement_trainer(self, **kwargs) -> Optional[ITrainer]:
        """Create reinforcement learning trainer"""
        return self.create("reinforcement", **kwargs)

class InferencerFactory(ComponentFactory[IInferencer]):
    """Factory for inference components"""
    
    def get_component_type(self) -> str:
        return "inferencer"
    
    def create_standard_inferencer(self, **kwargs) -> Optional[IInferencer]:
        """Create standard inferencer"""
        return self.create("standard", **kwargs)
    
    def create_batch_inferencer(self, **kwargs) -> Optional[IInferencer]:
        """Create batch inferencer"""
        return self.create("batch", **kwargs)
    
    def create_streaming_inferencer(self, **kwargs) -> Optional[IInferencer]:
        """Create streaming inferencer"""
        return self.create("streaming", **kwargs)
    
    def create_optimized_inferencer(self, **kwargs) -> Optional[IInferencer]:
        """Create optimized inferencer"""
        return self.create("optimized", **kwargs)

class MonitorFactory(ComponentFactory[IMonitor]):
    """Factory for monitoring components"""
    
    def get_component_type(self) -> str:
        return "monitor"
    
    def create_system_monitor(self, **kwargs) -> Optional[IMonitor]:
        """Create system monitor"""
        return self.create("system", **kwargs)
    
    def create_model_monitor(self, **kwargs) -> Optional[IMonitor]:
        """Create model monitor"""
        return self.create("model", **kwargs)
    
    def create_training_monitor(self, **kwargs) -> Optional[IMonitor]:
        """Create training monitor"""
        return self.create("training", **kwargs)
    
    def create_custom_monitor(self, metrics: List[str], **kwargs) -> Optional[IMonitor]:
        """Create custom monitor"""
        config = {"metrics": metrics, **kwargs}
        return self.create("custom", **config)

class BenchmarkerFactory(ComponentFactory[IBenchmarker]):
    """Factory for benchmarking components"""
    
    def get_component_type(self) -> str:
        return "benchmarker"
    
    def create_performance_benchmarker(self, **kwargs) -> Optional[IBenchmarker]:
        """Create performance benchmarker"""
        return self.create("performance", **kwargs)
    
    def create_memory_benchmarker(self, **kwargs) -> Optional[IBenchmarker]:
        """Create memory benchmarker"""
        return self.create("memory", **kwargs)
    
    def create_accuracy_benchmarker(self, **kwargs) -> Optional[IBenchmarker]:
        """Create accuracy benchmarker"""
        return self.create("accuracy", **kwargs)
    
    def create_comparative_benchmarker(self, **kwargs) -> Optional[IBenchmarker]:
        """Create comparative benchmarker"""
        return self.create("comparative", **kwargs)

class FactoryRegistry:
    """Registry for all factories"""
    
    def __init__(self):
        self.factories: Dict[str, ComponentFactory] = {}
        self._setup_default_factories()
    
    def _setup_default_factories(self):
        """Setup default factories"""
        self.factories["optimizer"] = OptimizerFactory()
        self.factories["model"] = ModelFactory()
        self.factories["trainer"] = TrainerFactory()
        self.factories["inferencer"] = InferencerFactory()
        self.factories["monitor"] = MonitorFactory()
        self.factories["benchmarker"] = BenchmarkerFactory()
    
    def get_factory(self, factory_type: str) -> Optional[ComponentFactory]:
        """Get factory by type"""
        return self.factories.get(factory_type)
    
    def register_factory(self, factory_type: str, factory: ComponentFactory) -> None:
        """Register a factory"""
        self.factories[factory_type] = factory
        logger.info(f"Registered factory: {factory_type}")
    
    def create_component(self, factory_type: str, component_name: str, **kwargs) -> Any:
        """Create component using factory"""
        factory = self.get_factory(factory_type)
        if factory:
            return factory.create(component_name, **kwargs)
        return None
    
    def list_factories(self) -> List[str]:
        """List available factories"""
        return list(self.factories.keys())
    
    def list_components(self, factory_type: str) -> List[str]:
        """List available components for a factory"""
        factory = self.get_factory(factory_type)
        if factory:
            return factory.list_available()
        return []

class FactoryBuilder:
    """Builder for complex component configurations"""
    
    def __init__(self, registry: FactoryRegistry):
        self.registry = registry
        self.configs: Dict[str, Dict[str, Any]] = {}
    
    def add_optimizer(self, name: str, level: str, **kwargs) -> 'FactoryBuilder':
        """Add optimizer configuration"""
        self.configs[f"optimizer_{name}"] = {
            "type": "optimizer",
            "name": name,
            "level": level,
            **kwargs
        }
        return self
    
    def add_model(self, name: str, model_type: str, **kwargs) -> 'FactoryBuilder':
        """Add model configuration"""
        self.configs[f"model_{name}"] = {
            "type": "model",
            "name": name,
            "model_type": model_type,
            **kwargs
        }
        return self
    
    def add_trainer(self, name: str, trainer_type: str, **kwargs) -> 'FactoryBuilder':
        """Add trainer configuration"""
        self.configs[f"trainer_{name}"] = {
            "type": "trainer",
            "name": name,
            "trainer_type": trainer_type,
            **kwargs
        }
        return self
    
    def add_inferencer(self, name: str, inferencer_type: str, **kwargs) -> 'FactoryBuilder':
        """Add inferencer configuration"""
        self.configs[f"inferencer_{name}"] = {
            "type": "inferencer",
            "name": name,
            "inferencer_type": inferencer_type,
            **kwargs
        }
        return self
    
    def add_monitor(self, name: str, monitor_type: str, **kwargs) -> 'FactoryBuilder':
        """Add monitor configuration"""
        self.configs[f"monitor_{name}"] = {
            "type": "monitor",
            "name": name,
            "monitor_type": monitor_type,
            **kwargs
        }
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build all configured components"""
        results = {}
        
        for config_name, config in self.configs.items():
            factory_type = config["type"]
            component_name = config["name"]
            
            # Remove type and name from config
            component_config = {k: v for k, v in config.items() if k not in ["type", "name"]}
            
            component = self.registry.create_component(factory_type, component_name, **component_config)
            if component:
                results[config_name] = component
        
        return results

class ComponentValidator:
    """Validator for component configurations"""
    
    def __init__(self):
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self._setup_default_schemas()
    
    def _setup_default_schemas(self):
        """Setup default validation schemas"""
        self.schemas["optimizer"] = {
            "required": ["level"],
            "optional": ["enable_adaptive_precision", "enable_memory_optimization"],
            "types": {
                "level": str,
                "enable_adaptive_precision": bool,
                "enable_memory_optimization": bool
            }
        }
        
        self.schemas["model"] = {
            "required": ["model_type"],
            "optional": ["hidden_size", "num_layers", "num_heads"],
            "types": {
                "model_type": str,
                "hidden_size": int,
                "num_layers": int,
                "num_heads": int
            }
        }
    
    def validate(self, component_type: str, config: Dict[str, Any]) -> bool:
        """Validate component configuration"""
        if component_type not in self.schemas:
            return True
        
        schema = self.schemas[component_type]
        
        # Check required fields
        for field in schema.get("required", []):
            if field not in config:
                logger.error(f"Missing required field {field} for {component_type}")
                return False
        
        # Check types
        for field, expected_type in schema.get("types", {}).items():
            if field in config and not isinstance(config[field], expected_type):
                logger.error(f"Invalid type for {field} in {component_type}: expected {expected_type}")
                return False
        
        return True
    
    def add_schema(self, component_type: str, schema: Dict[str, Any]) -> None:
        """Add validation schema"""
        self.schemas[component_type] = schema

