"""
Engine Factory for Blaze AI System.

This module provides a factory pattern implementation for creating
and managing different types of engines dynamically.
"""

from __future__ import annotations

import importlib
import inspect
from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Callable, Union
from pathlib import Path
import time

from .base import Engine, EngineType, EnginePriority
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from ..utils.logging import get_logger

# =============================================================================
# Engine Factory Configuration
# =============================================================================

@dataclass
class EngineFactoryConfig:
    """Configuration for the engine factory."""
    auto_discover_engines: bool = True
    discovery_paths: List[str] = None
    default_circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    enable_plugin_system: bool = True
    plugin_directory: str = "plugins"
    strict_validation: bool = True
    allow_custom_engines: bool = True
    
    def __post_init__(self):
        if self.discovery_paths is None:
            self.discovery_paths = ["engines"]
        if self.default_circuit_breaker_config is None:
            self.default_circuit_breaker_config = CircuitBreakerConfig()

@dataclass
class EngineTemplate:
    """Template for engine creation."""
    name: str
    engine_class: Type[Engine]
    default_config: Dict[str, Any]
    description: str = ""
    tags: List[str] = None
    priority: EnginePriority = EnginePriority.NORMAL
    dependencies: List[str] = None
    requirements: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.dependencies is None:
            self.dependencies = []
        if self.requirements is None:
            self.requirements = {}

# =============================================================================
# Engine Factory Implementation
# =============================================================================

class EngineFactory:
    """Factory for creating and managing engines."""
    
    def __init__(self, config: Optional[EngineFactoryConfig] = None):
        self.config = config or EngineFactoryConfig()
        self.logger = get_logger("engine_factory")
        self.engine_templates: Dict[str, EngineTemplate] = {}
        self.registered_engines: Dict[str, Engine] = {}
        self.engine_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Initialize factory
        self._initialize_factory()
    
    def _initialize_factory(self):
        """Initialize the engine factory."""
        self.logger.info("Initializing engine factory...")
        
        # Register built-in engines
        self._register_builtin_engines()
        
        # Auto-discover engines if enabled
        if self.config.auto_discover_engines:
            self._discover_engines()
        
        # Load plugins if enabled
        if self.config.enable_plugin_system:
            self._load_plugins()
        
        self.logger.info(f"Engine factory initialized with {len(self.engine_templates)} templates")
    
    def _register_builtin_engines(self):
        """Register built-in engine templates."""
        try:
            # LLM Engine
            from .llm import LLMEngine
            self.register_engine_template(
                "llm",
                LLMEngine,
                {
                    "model_name": "gpt2",
                    "cache_capacity": 1000,
                    "device": "auto",
                    "precision": "float16",
                    "enable_amp": True
                },
                "Large Language Model engine for text generation and processing",
                tags=["text", "nlp", "generation"],
                priority=EnginePriority.HIGH
            )
        except ImportError as e:
            self.logger.warning(f"Failed to register LLM engine: {e}")
        
        try:
            # Diffusion Engine
            from .diffusion import DiffusionEngine
            self.register_engine_template(
                "diffusion",
                DiffusionEngine,
                {
                    "model_id": "runwayml/stable-diffusion-v1-5",
                    "cache_capacity": 100,
                    "device": "auto",
                    "precision": "float16",
                    "enable_xformers": True
                },
                "Diffusion model engine for image generation",
                tags=["image", "generation", "diffusion"],
                priority=EnginePriority.HIGH
            )
        except ImportError as e:
            self.logger.warning(f"Failed to register Diffusion engine: {e}")
        
        try:
            # Router Engine
            from .router import RouterEngine
            self.register_engine_template(
                "router",
                RouterEngine,
                {
                    "enable_caching": True,
                    "cache_ttl": 1800,
                    "max_concurrent_requests": 50,
                    "load_balancing_strategy": "round_robin"
                },
                "Request routing and load balancing engine",
                tags=["routing", "load-balancing", "caching"],
                priority=EnginePriority.CRITICAL
            )
        except ImportError as e:
            self.logger.warning(f"Failed to register Router engine: {e}")
    
    def _discover_engines(self):
        """Auto-discover engines in specified paths."""
        for path in self.config.discovery_paths:
            try:
                self._discover_engines_in_path(path)
            except Exception as e:
                self.logger.warning(f"Failed to discover engines in {path}: {e}")
    
    def _discover_engines_in_path(self, path: str):
        """Discover engines in a specific path."""
        try:
            module = importlib.import_module(path)
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, Engine) and 
                    obj != Engine):
                    self._auto_register_discovered_engine(name, obj, path)
        except ImportError:
            self.logger.debug(f"Path {path} not importable, skipping")
    
    def _auto_register_discovered_engine(self, name: str, engine_class: Type[Engine], path: str):
        """Auto-register a discovered engine."""
        try:
            # Create default config based on engine type
            default_config = self._create_default_config_for_engine(engine_class)
            
            self.register_engine_template(
                name.lower(),
                engine_class,
                default_config,
                f"Auto-discovered {name} engine from {path}",
                tags=["auto-discovered", path],
                priority=EnginePriority.NORMAL
            )
            
            self.logger.info(f"Auto-registered engine: {name}")
        except Exception as e:
            self.logger.warning(f"Failed to auto-register engine {name}: {e}")
    
    def _create_default_config_for_engine(self, engine_class: Type[Engine]) -> Dict[str, Any]:
        """Create default configuration for an engine class."""
        # Try to get default config from class if available
        if hasattr(engine_class, 'get_default_config'):
            return engine_class.get_default_config()
        
        # Fallback to basic config
        return {
            "cache_capacity": 100,
            "device": "auto",
            "precision": "float16"
        }
    
    def _load_plugins(self):
        """Load engine plugins from plugin directory."""
        plugin_dir = Path(self.config.plugin_directory)
        if not plugin_dir.exists():
            return
        
        for plugin_file in plugin_dir.glob("*.py"):
            try:
                self._load_plugin(plugin_file)
            except Exception as e:
                self.logger.warning(f"Failed to load plugin {plugin_file}: {e}")
    
    def _load_plugin(self, plugin_file: Path):
        """Load a single plugin file."""
        try:
            spec = importlib.util.spec_from_file_location(plugin_file.stem, plugin_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Look for engine classes in the plugin
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, Engine) and 
                        obj != Engine):
                        self._register_plugin_engine(name, obj, plugin_file)
                        
        except Exception as e:
            self.logger.warning(f"Failed to load plugin {plugin_file}: {e}")
    
    def _register_plugin_engine(self, name: str, engine_class: Type[Engine], plugin_file: Path):
        """Register an engine from a plugin."""
        try:
            default_config = self._create_default_config_for_engine(engine_class)
            
            self.register_engine_template(
                f"plugin_{name.lower()}",
                engine_class,
                default_config,
                f"Plugin engine {name} from {plugin_file.name}",
                tags=["plugin", plugin_file.stem],
                priority=EnginePriority.LOW
            )
            
            self.logger.info(f"Registered plugin engine: {name}")
        except Exception as e:
            self.logger.warning(f"Failed to register plugin engine {name}: {e}")
    
    def register_engine_template(self, 
                               name: str, 
                               engine_class: Type[Engine], 
                               default_config: Dict[str, Any],
                               description: str = "",
                               tags: Optional[List[str]] = None,
                               priority: EnginePriority = EnginePriority.NORMAL,
                               dependencies: Optional[List[str]] = None,
                               requirements: Optional[Dict[str, Any]] = None) -> None:
        """Register an engine template."""
        if not issubclass(engine_class, Engine):
            raise ValueError(f"engine_class must be a subclass of Engine")
        
        template = EngineTemplate(
            name=name,
            engine_class=engine_class,
            default_config=default_config,
            description=description,
            tags=tags or [],
            priority=priority,
            dependencies=dependencies or [],
            requirements=requirements or {}
        )
        
        self.engine_templates[name] = template
        self.logger.info(f"Registered engine template: {name}")
    
    def create_engine(self, 
                     template_name: str, 
                     config: Optional[Dict[str, Any]] = None,
                     instance_name: Optional[str] = None) -> Engine:
        """Create an engine instance from a template."""
        if template_name not in self.engine_templates:
            raise ValueError(f"Engine template '{template_name}' not found")
        
        template = self.engine_templates[template_name]
        
        # Merge default config with provided config
        final_config = template.default_config.copy()
        if config:
            final_config.update(config)
        
        # Validate configuration
        if self.config.strict_validation:
            self._validate_engine_config(template, final_config)
        
        # Check dependencies
        self._check_engine_dependencies(template)
        
        # Create engine instance
        engine = template.engine_class(
            instance_name or template.name,
            final_config
        )
        
        # Store metadata
        engine_key = instance_name or template.name
        self.engine_metadata[engine_key] = {
            "template": template_name,
            "config": final_config,
            "created_at": time.time()
        }
        
        self.logger.info(f"Created engine instance: {engine_key} from template: {template_name}")
        return engine
    
    def _validate_engine_config(self, template: EngineTemplate, config: Dict[str, Any]):
        """Validate engine configuration."""
        # Basic validation - can be extended by subclasses
        if hasattr(template.engine_class, 'validate_config'):
            template.engine_class.validate_config(config)
    
    def _check_engine_dependencies(self, template: EngineTemplate):
        """Check if engine dependencies are satisfied."""
        for dependency in template.dependencies:
            try:
                importlib.import_module(dependency)
            except ImportError:
                raise ImportError(f"Engine {template.name} requires dependency: {dependency}")
    
    def get_available_templates(self) -> List[str]:
        """Get list of available engine template names."""
        return list(self.engine_templates.keys())
    
    def get_template_info(self, template_name: str) -> Optional[EngineTemplate]:
        """Get information about a specific template."""
        return self.engine_templates.get(template_name)
    
    def list_engines_by_type(self, engine_type: EngineType) -> List[str]:
        """List engines by type."""
        return [
            name for name, template in self.engine_templates.items()
            if hasattr(template.engine_class, '_get_engine_type') and
            template.engine_class._get_engine_type() == engine_type
        ]
    
    def list_engines_by_priority(self, priority: EnginePriority) -> List[str]:
        """List engines by priority."""
        return [
            name for name, template in self.engine_templates.items()
            if template.priority == priority
        ]
    
    def search_engines(self, query: str) -> List[str]:
        """Search engines by name, description, or tags."""
        query_lower = query.lower()
        results = []
        
        for name, template in self.engine_templates.items():
            if (query_lower in name.lower() or
                query_lower in template.description.lower() or
                any(query_lower in tag.lower() for tag in template.tags)):
                results.append(name)
        
        return results
    
    def get_engine_metadata(self, engine_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific engine instance."""
        return self.engine_metadata.get(engine_name)
    
    def unregister_template(self, template_name: str) -> bool:
        """Unregister an engine template."""
        if template_name in self.engine_templates:
            del self.engine_templates[template_name]
            self.logger.info(f"Unregistered engine template: {template_name}")
            return True
        return False
    
    def clear_templates(self):
        """Clear all engine templates."""
        self.engine_templates.clear()
        self.logger.info("Cleared all engine templates")

# =============================================================================
# Factory Functions
# =============================================================================

def create_engine_factory(config: Optional[EngineFactoryConfig] = None) -> EngineFactory:
    """Create an engine factory with custom configuration."""
    return EngineFactory(config)

def create_standard_engine_factory() -> EngineFactory:
    """Create a standard engine factory with default configuration."""
    config = EngineFactoryConfig(
        auto_discover_engines=True,
        enable_plugin_system=True,
        strict_validation=True
    )
    return EngineFactory(config)

def create_minimal_engine_factory() -> EngineFactory:
    """Create a minimal engine factory with basic configuration."""
    config = EngineFactoryConfig(
        auto_discover_engines=False,
        enable_plugin_system=False,
        strict_validation=False
    )
    return EngineFactory(config)

# =============================================================================
# Global Factory Instance
# =============================================================================

_default_factory: Optional[EngineFactory] = None

def get_engine_factory() -> EngineFactory:
    """Get the global engine factory instance."""
    global _default_factory
    if _default_factory is None:
        _default_factory = create_standard_engine_factory()
    return _default_factory

def set_engine_factory(factory: EngineFactory):
    """Set the global engine factory instance."""
    global _default_factory
    _default_factory = factory


