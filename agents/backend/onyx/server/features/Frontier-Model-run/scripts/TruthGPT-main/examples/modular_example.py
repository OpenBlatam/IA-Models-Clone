"""
Modular TruthGPT Example
Demonstrates the highly modular architecture with micro-modules, plugins, and dependency injection
"""

import torch
import torch.nn as nn
import logging
import time
from pathlib import Path

# Import the modular system
from core.modules import (
    BaseModule, ModuleManager, ModuleRegistry, ModuleFactory,
    IOptimizer, IModel, ITrainer, IInferencer, IMonitor, IBenchmarker,
    PluginManager, PluginLoader,
    OptimizerFactory, ModelFactory, TrainerFactory, InferencerFactory, MonitorFactory, BenchmarkerFactory,
    ConfigManager, ConfigValidator, ConfigBuilder,
    ComponentRegistry, ServiceRegistry, RegistryManager,
    DependencyInjector, ServiceContainer, AutoInjector, ServiceLocator
)
from core.modules.base import ModuleState
from core.modules.registry import ComponentType
from core.modules.injection import Scope

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example micro-modules
class MicroOptimizer(BaseModule, IOptimizer):
    """Micro optimization module"""
    
    def __init__(self, name: str, config: dict = None):
        super().__init__(name, config)
        self.optimization_level = config.get('level', 'basic')
        self.enabled_features = config.get('features', [])
    
    def initialize(self) -> bool:
        """Initialize the optimizer"""
        logger.info(f"Initializing micro optimizer: {self.name}")
        self.set_state(ModuleState.INITIALIZED)
        return True
    
    def start(self) -> bool:
        """Start the optimizer"""
        logger.info(f"Starting micro optimizer: {self.name}")
        self.set_state(ModuleState.RUNNING)
        return True
    
    def stop(self) -> bool:
        """Stop the optimizer"""
        logger.info(f"Stopping micro optimizer: {self.name}")
        self.set_state(ModuleState.STOPPED)
        return True
    
    def cleanup(self) -> bool:
        """Cleanup optimizer resources"""
        logger.info(f"Cleaning up micro optimizer: {self.name}")
        return True
    
    def optimize(self, model: nn.Module, **kwargs) -> nn.Module:
        """Optimize a model"""
        logger.info(f"Optimizing model with {self.name} (level: {self.optimization_level})")
        
        # Apply optimizations based on level
        if self.optimization_level == 'basic':
            return self._apply_basic_optimizations(model)
        elif self.optimization_level == 'enhanced':
            return self._apply_enhanced_optimizations(model)
        elif self.optimization_level == 'advanced':
            return self._apply_advanced_optimizations(model)
        else:
            return model
    
    def _apply_basic_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply basic optimizations"""
        model.eval()
        return model
    
    def _apply_enhanced_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply enhanced optimizations"""
        model.eval()
        # Add gradient checkpointing
        for module in model.modules():
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True
        return model
    
    def _apply_advanced_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply advanced optimizations"""
        model.eval()
        # Add gradient checkpointing and other optimizations
        for module in model.modules():
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True
        return model
    
    def get_optimization_info(self) -> dict:
        """Get optimization information"""
        return {
            "name": self.name,
            "level": self.optimization_level,
            "features": self.enabled_features,
            "state": self.get_state().value
        }
    
    def can_optimize(self, model: nn.Module) -> bool:
        """Check if model can be optimized"""
        return isinstance(model, nn.Module)

class MicroModel(BaseModule, IModel):
    """Micro model module"""
    
    def __init__(self, name: str, config: dict = None):
        super().__init__(name, config)
        self.model_type = config.get('model_type', 'transformer')
        self.model = None
    
    def initialize(self) -> bool:
        """Initialize the model"""
        logger.info(f"Initializing micro model: {self.name}")
        self.set_state(ModuleState.INITIALIZED)
        return True
    
    def start(self) -> bool:
        """Start the model"""
        logger.info(f"Starting micro model: {self.name}")
        self.set_state(ModuleState.RUNNING)
        return True
    
    def stop(self) -> bool:
        """Stop the model"""
        logger.info(f"Stopping micro model: {self.name}")
        self.set_state(ModuleState.STOPPED)
        return True
    
    def cleanup(self) -> bool:
        """Cleanup model resources"""
        logger.info(f"Cleaning up micro model: {self.name}")
        return True
    
    def load(self, path: str = None) -> nn.Module:
        """Load a model"""
        if path:
            logger.info(f"Loading model from {path}")
            # In real implementation, load from file
            self.model = self._create_model()
        else:
            self.model = self._create_model()
        return self.model
    
    def save(self, model: nn.Module, path: str) -> bool:
        """Save a model"""
        logger.info(f"Saving model to {path}")
        # In real implementation, save to file
        return True
    
    def create(self, config: dict) -> nn.Module:
        """Create a new model"""
        logger.info(f"Creating model with config: {config}")
        return self._create_model()
    
    def get_info(self, model: nn.Module) -> dict:
        """Get model information"""
        return {
            "name": self.name,
            "type": self.model_type,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
    
    def _create_model(self) -> nn.Module:
        """Create a simple model"""
        if self.model_type == 'transformer':
            return nn.Sequential(
                nn.Linear(100, 50),
                nn.ReLU(),
                nn.Linear(50, 10)
            )
        else:
            return nn.Linear(10, 5)

class MicroTrainer(BaseModule, ITrainer):
    """Micro trainer module"""
    
    def __init__(self, name: str, config: dict = None):
        super().__init__(name, config)
        self.epochs = config.get('epochs', 10)
        self.batch_size = config.get('batch_size', 32)
        self.learning_rate = config.get('learning_rate', 1e-3)
    
    def initialize(self) -> bool:
        """Initialize the trainer"""
        logger.info(f"Initializing micro trainer: {self.name}")
        self.set_state(ModuleState.INITIALIZED)
        return True
    
    def start(self) -> bool:
        """Start the trainer"""
        logger.info(f"Starting micro trainer: {self.name}")
        self.set_state(ModuleState.RUNNING)
        return True
    
    def stop(self) -> bool:
        """Stop the trainer"""
        logger.info(f"Stopping micro trainer: {self.name}")
        self.set_state(ModuleState.STOPPED)
        return True
    
    def cleanup(self) -> bool:
        """Cleanup trainer resources"""
        logger.info(f"Cleaning up micro trainer: {self.name}")
        return True
    
    def setup(self, model: nn.Module, train_data: any, val_data: any = None) -> bool:
        """Setup training"""
        logger.info(f"Setting up training for {self.name}")
        return True
    
    def train(self, **kwargs) -> dict:
        """Train the model"""
        logger.info(f"Training with {self.name} for {self.epochs} epochs")
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "status": "completed"
        }
    
    def validate(self, model: nn.Module, data: any) -> dict:
        """Validate the model"""
        logger.info(f"Validating with {self.name}")
        return {"accuracy": 0.95, "loss": 0.1}
    
    def save_checkpoint(self, path: str) -> bool:
        """Save training checkpoint"""
        logger.info(f"Saving checkpoint to {path}")
        return True
    
    def load_checkpoint(self, path: str) -> bool:
        """Load training checkpoint"""
        logger.info(f"Loading checkpoint from {path}")
        return True

class MicroInferencer(BaseModule, IInferencer):
    """Micro inferencer module"""
    
    def __init__(self, name: str, config: dict = None):
        super().__init__(name, config)
        self.model = None
        self.tokenizer = None
    
    def initialize(self) -> bool:
        """Initialize the inferencer"""
        logger.info(f"Initializing micro inferencer: {self.name}")
        self.set_state(ModuleState.INITIALIZED)
        return True
    
    def start(self) -> bool:
        """Start the inferencer"""
        logger.info(f"Starting micro inferencer: {self.name}")
        self.set_state(ModuleState.RUNNING)
        return True
    
    def stop(self) -> bool:
        """Stop the inferencer"""
        logger.info(f"Stopping micro inferencer: {self.name}")
        self.set_state(ModuleState.STOPPED)
        return True
    
    def cleanup(self) -> bool:
        """Cleanup inferencer resources"""
        logger.info(f"Cleaning up micro inferencer: {self.name}")
        return True
    
    def load_model(self, model: nn.Module, tokenizer: any = None) -> None:
        """Load model for inference"""
        self.model = model
        self.tokenizer = tokenizer
        logger.info(f"Loaded model for {self.name}")
    
    def generate(self, input_data: str, **kwargs) -> dict:
        """Generate output"""
        logger.info(f"Generating with {self.name}")
        return {
            "input": input_data,
            "output": f"Generated by {self.name}",
            "tokens": 10
        }
    
    def batch_generate(self, inputs: list, **kwargs) -> list:
        """Generate for multiple inputs"""
        logger.info(f"Batch generating with {self.name}")
        return [self.generate(input_data) for input_data in inputs]
    
    def optimize_for_inference(self) -> None:
        """Optimize for inference"""
        logger.info(f"Optimizing {self.name} for inference")
        if self.model:
            self.model.eval()

class MicroMonitor(BaseModule, IMonitor):
    """Micro monitor module"""
    
    def __init__(self, name: str, config: dict = None):
        super().__init__(name, config)
        self.metrics = {}
        self.monitoring = False
    
    def initialize(self) -> bool:
        """Initialize the monitor"""
        logger.info(f"Initializing micro monitor: {self.name}")
        self.set_state(ModuleState.INITIALIZED)
        return True
    
    def start(self) -> bool:
        """Start the monitor"""
        logger.info(f"Starting micro monitor: {self.name}")
        self.set_state(ModuleState.RUNNING)
        return True
    
    def stop(self) -> bool:
        """Stop the monitor"""
        logger.info(f"Stopping micro monitor: {self.name}")
        self.set_state(ModuleState.STOPPED)
        return True
    
    def cleanup(self) -> bool:
        """Cleanup monitor resources"""
        logger.info(f"Cleaning up micro monitor: {self.name}")
        return True
    
    def start_monitoring(self, interval: float = 1.0) -> None:
        """Start monitoring"""
        self.monitoring = True
        logger.info(f"Started monitoring with {self.name} (interval: {interval}s)")
    
    def stop_monitoring(self) -> None:
        """Stop monitoring"""
        self.monitoring = False
        logger.info(f"Stopped monitoring with {self.name}")
    
    def record_metric(self, name: str, value: float, metadata: dict = None) -> None:
        """Record a metric"""
        self.metrics[name] = {
            "value": value,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        logger.info(f"Recorded metric {name}: {value}")
    
    def get_metrics(self) -> dict:
        """Get all metrics"""
        return self.metrics
    
    def get_report(self) -> dict:
        """Get comprehensive report"""
        return {
            "name": self.name,
            "monitoring": self.monitoring,
            "metrics_count": len(self.metrics),
            "metrics": self.metrics
        }

def demonstrate_modular_architecture():
    """Demonstrate the modular architecture"""
    logger.info("🏗️ Demonstrating Modular TruthGPT Architecture")
    logger.info("=" * 60)
    
    # 1. Module Management
    logger.info("\n📦 PHASE 1: Module Management")
    module_manager = ModuleManager()
    
    # Register module classes
    module_manager.register_module_class("micro_optimizer", MicroOptimizer)
    module_manager.register_module_class("micro_model", MicroModel)
    module_manager.register_module_class("micro_trainer", MicroTrainer)
    module_manager.register_module_class("micro_inferencer", MicroInferencer)
    module_manager.register_module_class("micro_monitor", MicroMonitor)
    
    # Create modules
    optimizer = module_manager.create_module("optimizer_1", "micro_optimizer", 
                                           level="enhanced", features=["memory", "precision"])
    model = module_manager.create_module("model_1", "micro_model", 
                                       model_type="transformer", hidden_size=128)
    trainer = module_manager.create_module("trainer_1", "micro_trainer", 
                                         epochs=5, batch_size=16)
    inferencer = module_manager.create_module("inferencer_1", "micro_inferencer")
    monitor = module_manager.create_module("monitor_1", "micro_monitor")
    
    # Start modules
    logger.info("🚀 Starting modules...")
    start_results = module_manager.start_all()
    logger.info(f"Start results: {start_results}")
    
    # 2. Plugin System
    logger.info("\n🔌 PHASE 2: Plugin System")
    plugin_manager = PluginManager()
    
    # Add plugin directory (would be a real directory in practice)
    plugin_manager.add_plugin_directory("./plugins")
    
    # Discover and load plugins
    discovered = plugin_manager.discover_plugins()
    logger.info(f"Discovered plugins: {discovered}")
    
    # 3. Factory System
    logger.info("\n🏭 PHASE 3: Factory System")
    registry = FactoryRegistry()
    
    # Register components with factories
    registry.get_factory("optimizer").register("micro", MicroOptimizer)
    registry.get_factory("model").register("micro", MicroModel)
    registry.get_factory("trainer").register("micro", MicroTrainer)
    registry.get_factory("inferencer").register("micro", MicroInferencer)
    registry.get_factory("monitor").register("micro", MicroMonitor)
    
    # Create components using factories
    factory_optimizer = registry.create_component("optimizer", "micro", level="advanced")
    factory_model = registry.create_component("model", "micro", model_type="transformer")
    
    logger.info(f"Factory created components: {factory_optimizer is not None}, {factory_model is not None}")
    
    # 4. Configuration Management
    logger.info("\n⚙️ PHASE 4: Configuration Management")
    config_manager = ConfigManager()
    
    # Add configuration sources
    from core.modules.config import ConfigSource, ConfigFormat
    config_source = ConfigSource("app_config", "config.json", ConfigFormat.JSON, priority=1)
    config_manager.add_source(config_source)
    
    # Load configurations
    load_results = config_manager.load_all()
    logger.info(f"Config load results: {load_results}")
    
    # 5. Registry System
    logger.info("\n📋 PHASE 5: Registry System")
    registry_manager = RegistryManager()
    
    # Register components
    registry_manager.register_component("optimizer_1", ComponentType.OPTIMIZER, MicroOptimizer)
    registry_manager.register_component("model_1", ComponentType.MODEL, MicroModel)
    registry_manager.register_component("trainer_1", ComponentType.TRAINER, MicroTrainer)
    
    # Get components
    registered_optimizer = registry_manager.get_component("optimizer_1")
    registered_model = registry_manager.get_component("model_1")
    
    logger.info(f"Registry components: {registered_optimizer is not None}, {registered_model is not None}")
    
    # 6. Dependency Injection
    logger.info("\n💉 PHASE 6: Dependency Injection")
    injector = DependencyInjector()
    
    # Register services
    injector.register_singleton(IOptimizer, implementation_type=MicroOptimizer)
    injector.register_singleton(IModel, implementation_type=MicroModel)
    injector.register_transient(ITrainer, implementation_type=MicroTrainer)
    
    # Create scope
    scope_id = injector.create_scope()
    
    # Get services with dependency injection
    injected_optimizer = injector.get(IOptimizer, scope_id)
    injected_model = injector.get(IModel, scope_id)
    injected_trainer = injector.get(ITrainer, scope_id)
    
    logger.info(f"Injected services: {injected_optimizer is not None}, {injected_model is not None}, {injected_trainer is not None}")
    
    # 7. Complete Workflow
    logger.info("\n🔄 PHASE 7: Complete Modular Workflow")
    
    # Create a simple model
    test_model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    # Optimize model
    if optimizer:
        optimized_model = optimizer.optimize(test_model)
        logger.info(f"Model optimized: {optimized_model is not None}")
    
    # Train model
    if trainer:
        trainer.setup(test_model, None, None)
        training_results = trainer.train()
        logger.info(f"Training results: {training_results}")
    
    # Inference
    if inferencer:
        inferencer.load_model(test_model)
        result = inferencer.generate("Hello, world!")
        logger.info(f"Inference result: {result}")
    
    # Monitoring
    if monitor:
        monitor.start_monitoring()
        monitor.record_metric("accuracy", 0.95, {"epoch": 1})
        monitor.record_metric("loss", 0.1, {"epoch": 1})
        report = monitor.get_report()
        logger.info(f"Monitor report: {report}")
    
    # 8. Cleanup
    logger.info("\n🧹 PHASE 8: Cleanup")
    
    # Stop modules
    stop_results = module_manager.stop_all()
    logger.info(f"Stop results: {stop_results}")
    
    # Dispose scope
    injector.dispose_scope(scope_id)
    
    logger.info("✅ Modular architecture demonstration completed!")

def demonstrate_micro_modules():
    """Demonstrate individual micro-modules"""
    logger.info("\n🔬 Demonstrating Individual Micro-Modules")
    
    # Create micro-modules
    modules = {
        "optimizer": MicroOptimizer("micro_opt", {"level": "enhanced", "features": ["memory", "precision"]}),
        "model": MicroModel("micro_model", {"model_type": "transformer", "hidden_size": 256}),
        "trainer": MicroTrainer("micro_trainer", {"epochs": 10, "batch_size": 32, "learning_rate": 1e-3}),
        "inferencer": MicroInferencer("micro_inferencer"),
        "monitor": MicroMonitor("micro_monitor")
    }
    
    # Initialize all modules
    for name, module in modules.items():
        module.initialize()
        logger.info(f"✅ Initialized {name}: {module.get_state().value}")
    
    # Start all modules
    for name, module in modules.items():
        module.start()
        logger.info(f"✅ Started {name}: {module.get_state().value}")
    
    # Demonstrate functionality
    test_model = nn.Linear(10, 5)
    
    # Optimize
    optimized = modules["optimizer"].optimize(test_model)
    logger.info(f"✅ Model optimized: {optimized is not None}")
    
    # Train
    modules["trainer"].setup(test_model, None, None)
    training_results = modules["trainer"].train()
    logger.info(f"✅ Training completed: {training_results}")
    
    # Inference
    modules["inferencer"].load_model(test_model)
    result = modules["inferencer"].generate("Test input")
    logger.info(f"✅ Inference completed: {result}")
    
    # Monitor
    modules["monitor"].start_monitoring()
    modules["monitor"].record_metric("test_metric", 42.0)
    metrics = modules["monitor"].get_metrics()
    logger.info(f"✅ Monitoring: {len(metrics)} metrics recorded")
    
    # Stop all modules
    for name, module in modules.items():
        module.stop()
        logger.info(f"✅ Stopped {name}: {module.get_state().value}")

def demonstrate_plugin_system():
    """Demonstrate plugin system"""
    logger.info("\n🔌 Demonstrating Plugin System")
    
    # Create plugin manager
    plugin_manager = PluginManager()
    
    # Add plugin directory
    plugin_manager.add_plugin_directory("./plugins")
    
    # Discover plugins
    discovered = plugin_manager.discover_plugins()
    logger.info(f"Discovered plugins: {discovered}")
    
    # Load plugins
    plugin_results = plugin_manager.discover_and_load_plugins()
    logger.info(f"Plugin load results: {plugin_results}")
    
    # Get plugin status
    status = plugin_manager.get_plugin_status()
    logger.info(f"Plugin status: {status}")

def demonstrate_factory_patterns():
    """Demonstrate factory patterns"""
    logger.info("\n🏭 Demonstrating Factory Patterns")
    
    # Create factory registry
    registry = FactoryRegistry()
    
    # Register components
    registry.get_factory("optimizer").register("micro", MicroOptimizer)
    registry.get_factory("model").register("micro", MicroModel)
    registry.get_factory("trainer").register("micro", MicroTrainer)
    
    # Create components
    optimizer = registry.create_component("optimizer", "micro", level="advanced")
    model = registry.create_component("model", "micro", model_type="transformer")
    trainer = registry.create_component("trainer", "micro", epochs=5)
    
    logger.info(f"Factory created: optimizer={optimizer is not None}, model={model is not None}, trainer={trainer is not None}")
    
    # List available components
    for factory_type in registry.list_factories():
        components = registry.list_components(factory_type)
        logger.info(f"Factory {factory_type}: {components}")

def demonstrate_dependency_injection():
    """Demonstrate dependency injection"""
    logger.info("\n💉 Demonstrating Dependency Injection")
    
    # Create injector
    injector = DependencyInjector()
    
    # Register services
    injector.register_singleton(IOptimizer, implementation_type=MicroOptimizer)
    injector.register_transient(ITrainer, implementation_type=MicroTrainer)
    injector.register_scoped(IModel, implementation_type=MicroModel)
    
    # Create scope
    scope_id = injector.create_scope()
    
    # Get services
    optimizer = injector.get(IOptimizer, scope_id)
    trainer = injector.get(ITrainer, scope_id)
    model = injector.get(IModel, scope_id)
    
    logger.info(f"Injected services: optimizer={optimizer is not None}, trainer={trainer is not None}, model={model is not None}")
    
    # Create service locator
    locator = ServiceLocator(injector)
    
    # Get services through locator
    locator_optimizer = locator.get(IOptimizer, scope_id)
    logger.info(f"Service locator: {locator_optimizer is not None}")
    
    # Cleanup
    injector.dispose_scope(scope_id)

def main():
    """Main demonstration function"""
    logger.info("🎉 Starting Modular TruthGPT Demonstration")
    logger.info("=" * 60)
    
    try:
        # Demonstrate modular architecture
        demonstrate_modular_architecture()
        
        # Demonstrate individual components
        demonstrate_micro_modules()
        
        # Demonstrate plugin system
        demonstrate_plugin_system()
        
        # Demonstrate factory patterns
        demonstrate_factory_patterns()
        
        # Demonstrate dependency injection
        demonstrate_dependency_injection()
        
        logger.info("\n🎊 Modular TruthGPT Demonstration Completed Successfully!")
        
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        raise

if __name__ == "__main__":
    main()

