"""
Core modules for modular architecture.
"""
from .config import ConfigManager, TrainerConfig
from .interfaces import (
    BaseTrainer,
    BaseEvaluator,
    BaseModelManager,
    BaseDataLoader,
    BaseCheckpointManager,
)
from .service_registry import (
    ServiceRegistry,
    ServiceContainer,
    register_service,
    get_service,
)
from .event_system import (
    EventEmitter,
    EventType,
    Event,
    get_event_emitter,
    emit_event,
    on_event,
)
from .plugin_system import (
    Plugin,
    PluginManager,
    get_plugin_manager,
)
from .dynamic_factory import (
    DynamicFactory,
    factory,
    register_component,
    create_factory,
)
from .composition import (
    ComponentAssembler,
    WorkflowBuilder,
)
from .validation import (
    Validator,
    ModelValidator,
    DataValidator,
    ConfigValidator,
)
from .module_loader import (
    ModuleLoader,
    get_module_loader,
    lazy_load,
)

__all__ = [
    # Config
    "ConfigManager",
    "TrainerConfig",
    # Interfaces
    "BaseTrainer",
    "BaseEvaluator",
    "BaseModelManager",
    "BaseDataLoader",
    "BaseCheckpointManager",
    # Service Registry
    "ServiceRegistry",
    "ServiceContainer",
    "register_service",
    "get_service",
    # Event System
    "EventEmitter",
    "EventType",
    "Event",
    "get_event_emitter",
    "emit_event",
    "on_event",
    # Plugin System
    "Plugin",
    "PluginManager",
    "get_plugin_manager",
    # Dynamic Factory
    "DynamicFactory",
    "factory",
    "register_component",
    "create_factory",
    # Composition
    "ComponentAssembler",
    "WorkflowBuilder",
    # Validation
    "Validator",
    "ModelValidator",
    "DataValidator",
    "ConfigValidator",
    # Module Loader
    "ModuleLoader",
    "get_module_loader",
    "lazy_load",
]
