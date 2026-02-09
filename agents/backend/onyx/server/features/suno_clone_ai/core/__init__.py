"""
Core Module - Arquitectura Modular

Proporciona interfaces, factories, DI, eventos y plugins.
"""

from .interfaces import (
    IMusicGenerator,
    IAudioProcessor,
    ICacheManager,
    IStorageBackend,
    INotificationService,
    IAnalyticsService,
    IAuthenticationService,
    IPlugin
)

from .factories import (
    ServiceFactory,
    MusicGeneratorFactory,
    CacheFactory,
    StorageFactory,
    NotificationFactory
)

from .dependency_injection import (
    DependencyContainer,
    get_container,
    inject,
    resolve_dependency
)

from .events import (
    EventType,
    Event,
    EventBus,
    get_event_bus,
    event_handler
)

from .storage import LocalStorage

from .plugins import (
    PluginManager,
    get_plugin_manager,
    BasePlugin
)

from .modules import (
    ModuleRegistry,
    get_module_registry,
    BaseModule
)

from .service_locator import (
    ServiceLocator,
    get_service,
    resolve_service
)

from .validators import (
    Validator,
    ValidationError,
    validate_and_raise
)

from .helpers import (
    generate_id,
    hash_string,
    safe_json_loads,
    safe_json_dumps,
    format_duration,
    format_file_size,
    ensure_directory,
    chunk_list,
    merge_dicts,
    get_nested_value,
    set_nested_value,
    sanitize_filename,
    retry_on_failure
)

from .initialization import (
    SystemInitializer,
    get_system_initializer,
    initialize_system
)

__all__ = [
    # Interfaces
    "IMusicGenerator",
    "IAudioProcessor",
    "ICacheManager",
    "IStorageBackend",
    "INotificationService",
    "IAnalyticsService",
    "IAuthenticationService",
    "IPlugin",
    # Factories
    "ServiceFactory",
    "MusicGeneratorFactory",
    "CacheFactory",
    "StorageFactory",
    "NotificationFactory",
    # Dependency Injection
    "DependencyContainer",
    "get_container",
    "inject",
    "resolve_dependency",
    # Events
    "EventType",
    "Event",
    "EventBus",
    "get_event_bus",
    "event_handler",
    # Storage
    "LocalStorage",
    # Plugins
    "PluginManager",
    "get_plugin_manager",
    "BasePlugin",
    # Modules
    "ModuleRegistry",
    "get_module_registry",
    "BaseModule",
    # Service Locator
    "ServiceLocator",
    "get_service",
    "resolve_service",
    # Validators
    "Validator",
    "ValidationError",
    "validate_and_raise",
    # Helpers
    "generate_id",
    "hash_string",
    "safe_json_loads",
    "safe_json_dumps",
    "format_duration",
    "format_file_size",
    "ensure_directory",
    "chunk_list",
    "merge_dicts",
    "get_nested_value",
    "set_nested_value",
    "sanitize_filename",
    "retry_on_failure",
    # Initialization
    "SystemInitializer",
    "get_system_initializer",
    "initialize_system"
]
