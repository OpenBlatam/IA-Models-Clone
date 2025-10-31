"""
🚀 BLATAM AI ENGINES MODULE v6.0.0
==================================

Módulo de motores AI organizados y modulares:
- ⚡ Speed Engine (Ultra-fast optimizations)
- 🧠 NLP Engine (Advanced language processing)
- 🔗 LangChain Engine (Intelligent orchestration)
- 🔄 Evolution Engine (Self-improving system)
- 🎯 Multi-Modal Engine (Cross-modal processing)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

from ..core import BlatamComponent, ComponentConfig, ComponentFactory, ServiceContainer

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# 🎯 ENGINE CONFIGURATION
# =============================================================================

@dataclass
class EngineConfig:
    """Base configuration for all engines."""
    name: str
    enabled: bool = True
    max_workers: int = 4
    timeout: float = 30.0
    retry_attempts: int = 3
    cache_size: int = 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'max_workers': self.max_workers,
            'timeout': self.timeout,
            'retry_attempts': self.retry_attempts,
            'cache_size': self.cache_size
        }

# =============================================================================
# 🏭 ENGINE FACTORY REGISTRY
# =============================================================================

class EngineRegistry:
    """Centralized registry for available engines."""
    
    def __init__(self) -> None:
        self._engine_factories: Dict[str, ComponentFactory] = {}
        self._engine_configs: Dict[str, type] = {}
        self._engine_dependencies: Dict[str, List[str]] = {}
        self._engine_metadata: Dict[str, Dict[str, Any]] = {}
    
    def register_engine(
        self,
        engine_type: str,
        factory: ComponentFactory,
        config_class: type,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register an engine with its factory and configuration."""
        self._engine_factories[engine_type] = factory
        self._engine_configs[engine_type] = config_class
        self._engine_dependencies[engine_type] = dependencies or []
        self._engine_metadata[engine_type] = metadata or {}
        logger.info(f"🔧 Registered engine: {engine_type}")
    
    def get_available_engines(self) -> List[str]:
        """Get list of available engines."""
        return list(self._engine_factories.keys())
    
    def get_engine_factory(self, engine_type: str) -> ComponentFactory:
        """Get factory for a specific engine."""
        if engine_type not in self._engine_factories:
            raise ValueError(f"Engine '{engine_type}' not registered")
        return self._engine_factories[engine_type]
    
    def get_engine_config_class(self, engine_type: str) -> type:
        """Get configuration class for a specific engine."""
        if engine_type not in self._engine_configs:
            raise ValueError(f"Engine config for '{engine_type}' not found")
        return self._engine_configs[engine_type]
    
    def get_engine_dependencies(self, engine_type: str) -> List[str]:
        """Get dependencies for a specific engine."""
        return self._engine_dependencies.get(engine_type, [])
    
    def get_engine_metadata(self, engine_type: str) -> Dict[str, Any]:
        """Get metadata for a specific engine."""
        return self._engine_metadata.get(engine_type, {})
    
    def resolve_dependency_order(self, engine_types: List[str]) -> List[str]:
        """Resolve dependency order for engine initialization."""
        resolved = []
        pending = set(engine_types)
        
        while pending:
            ready = []
            for engine_type in pending:
                dependencies = self.get_engine_dependencies(engine_type)
                if all(dep in resolved for dep in dependencies):
                    ready.append(engine_type)
            
            if not ready:
                raise ValueError(f"Circular dependency detected in engines: {pending}")
            
            for engine_type in ready:
                resolved.append(engine_type)
                pending.remove(engine_type)
        
        return resolved
    
    def validate_engine_registration(self, engine_type: str) -> bool:
        """Validate that an engine is properly registered."""
        return (
            engine_type in self._engine_factories and
            engine_type in self._engine_configs and
            engine_type in self._engine_dependencies
        )

# =============================================================================
# 🎯 ENGINE MANAGER
# =============================================================================

class EngineManager:
    """Centralized engine manager with improved error handling."""
    
    def __init__(self, service_container: ServiceContainer):
        self.service_container = service_container
        self.registry = EngineRegistry()
        self.engines: Dict[str, BlatamComponent] = {}
        self.engine_configs: Dict[str, Any] = {}
        self.is_initialized = False
        self._initialization_lock = asyncio.Lock()
    
    async def initialize_engines(
        self,
        engine_configs: Dict[str, Dict[str, Any]],
        enabled_engines: Optional[List[str]] = None
    ) -> bool:
        """Initialize specified engines with proper error handling."""
        async with self._initialization_lock:
            if self.is_initialized:
                logger.warning("⚠️ Engines already initialized")
                return True
                
            try:
                logger.info("🚀 Initializing engines...")
                
                # Determine which engines to initialize
                if enabled_engines is None:
                    enabled_engines = list(engine_configs.keys())
                
                # Validate engine configurations
                if not self._validate_engine_configs(engine_configs, enabled_engines):
                    return False
                
                # Resolve dependency order
                ordered_engines = self.registry.resolve_dependency_order(enabled_engines)
                
                # Initialize engines in order
                for engine_type in ordered_engines:
                    if engine_type in engine_configs:
                        success = await self._initialize_engine(engine_type, engine_configs[engine_type])
                        if not success:
                            logger.error(f"❌ Failed to initialize engine: {engine_type}")
                            await self._cleanup_failed_initialization()
                            return False
                
                self.is_initialized = True
                logger.info(f"✅ Engines initialized successfully: {list(self.engines.keys())}")
                return True
                
            except Exception as e:
                logger.error(f"❌ Engine initialization failed: {e}")
                await self._cleanup_failed_initialization()
                return False
    
    def _validate_engine_configs(
        self, 
        engine_configs: Dict[str, Dict[str, Any]], 
        enabled_engines: List[str]
    ) -> bool:
        """Validate engine configurations before initialization."""
        for engine_type in enabled_engines:
            if engine_type not in engine_configs:
                logger.error(f"❌ Configuration missing for engine: {engine_type}")
                return False
            
            if not self.registry.validate_engine_registration(engine_type):
                logger.error(f"❌ Engine not properly registered: {engine_type}")
                return False
        
        return True
    
    async def _initialize_engine(self, engine_type: str, config: Dict[str, Any]) -> bool:
        """Initialize a specific engine with improved error handling."""
        try:
            # Get factory and configuration
            factory = self.registry.get_engine_factory(engine_type)
            config_class = self.registry.get_engine_config_class(engine_type)
            
            # Create typed configuration
            try:
                if hasattr(config_class, 'from_dict'):
                    typed_config = config_class.from_dict(config)
                else:
                    typed_config = config_class(**config)
            except Exception as e:
                logger.error(f"❌ Failed to create config for engine '{engine_type}': {e}")
                return False
            
            # Create engine instance
            try:
                engine = await factory.create_component(
                    typed_config, 
                    service_container=self.service_container
                )
            except Exception as e:
                logger.error(f"❌ Failed to create engine '{engine_type}': {e}")
                return False
            
            # Initialize engine
            try:
                success = await engine.initialize()
                if success:
                    self.engines[engine_type] = engine
                    self.engine_configs[engine_type] = typed_config
                    
                    # Register in service container
                    self.service_container.register_service(f"{engine_type}_engine", engine)
                    
                    logger.info(f"✅ Engine '{engine_type}' initialized successfully")
                    return True
                else:
                    logger.error(f"❌ Engine '{engine_type}' initialization failed")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Error during engine '{engine_type}' initialization: {e}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Unexpected error initializing engine '{engine_type}': {e}")
            return False
    
    async def _cleanup_failed_initialization(self) -> None:
        """Cleanup resources after failed initialization."""
        for engine_type, engine in self.engines.items():
            try:
                if hasattr(engine, 'shutdown'):
                    await engine.shutdown()
                logger.info(f"🧹 Cleaned up engine: {engine_type}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to cleanup engine '{engine_type}': {e}")
        
        self.engines.clear()
        self.engine_configs.clear()
        self.is_initialized = False
    
    def get_engine(self, engine_type: str) -> Optional[BlatamComponent]:
        """Get a specific engine."""
        if not self.is_initialized:
            logger.warning("⚠️ Engines not yet initialized")
            return None
        return self.engines.get(engine_type)
    
    def get_all_engines(self) -> Dict[str, BlatamComponent]:
        """Get all initialized engines."""
        if not self.is_initialized:
            logger.warning("⚠️ Engines not yet initialized")
            return {}
        return self.engines.copy()
    
    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Check health of all engines."""
        if not self.is_initialized:
            return {"status": "not_initialized"}
        
        health_results = {}
        
        for engine_type, engine in self.engines.items():
            try:
                if hasattr(engine, 'health_check'):
                    health_results[engine_type] = await engine.health_check()
                else:
                    health_results[engine_type] = {'status': 'no_health_check_method'}
            except Exception as e:
                health_results[engine_type] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return health_results
    
    def get_stats_all(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics from all engines."""
        if not self.is_initialized:
            return {"status": "not_initialized"}
        
        stats = {}
        
        for engine_type, engine in self.engines.items():
            try:
                if hasattr(engine, 'get_stats'):
                    stats[engine_type] = engine.get_stats()
                else:
                    stats[engine_type] = {'status': 'no_stats_method'}
            except Exception as e:
                stats[engine_type] = {
                    'error': str(e)
                }
        
        return stats
    
    async def shutdown_all(self) -> None:
        """Shutdown all engines gracefully."""
        if not self.is_initialized:
            return
        
        logger.info("🔄 Shutting down all engines...")
        
        for engine_type, engine in self.engines.items():
            try:
                if hasattr(engine, 'shutdown'):
                    await engine.shutdown()
                logger.info(f"✅ Engine '{engine_type}' shutdown successfully")
            except Exception as e:
                logger.warning(f"⚠️ Failed to shutdown engine '{engine_type}': {e}")
        
        self.engines.clear()
        self.engine_configs.clear()
        self.is_initialized = False
        logger.info("✅ All engines shutdown complete")

# =============================================================================
# 🔧 ENGINE INITIALIZATION HELPERS
# =============================================================================

def create_default_engine_configs() -> Dict[str, Dict[str, Any]]:
    """Create default configurations for all engines."""
    return {
        'speed': {
            'enable_uvloop': True,
            'enable_fast_cache': True,
            'enable_lazy_loading': True,
            'enable_worker_pool': True,
            'cache_size': 10000,
            'max_workers': 8
        },
        'nlp': {
            'primary_llm': 'gpt-4-turbo-preview',
            'embedding_model': 'text-embedding-3-large',
            'enable_multilingual': True,
            'enable_speech': True,
            'enable_sentiment': True,
            'enable_entities': True
        },
        'langchain': {
            'llm_provider': 'openai',
            'llm_model': 'gpt-4-turbo-preview',
            'default_agent_type': 'openai-functions',
            'enable_web_search': True,
            'enable_python_repl': True,
            'vector_store_type': 'chroma'
        },
        'evolution': {
            'optimization_strategy': 'balanced',
            'learning_mode': 'active',
            'auto_optimization_interval': 300,
            'enable_self_healing': True,
            'enable_predictive_scaling': True,
            'enable_continuous_learning': True,
            'enable_multi_modal': True
        }
    }

async def create_optimized_engine_manager(
    service_container: Optional[ServiceContainer] = None,
    custom_configs: Optional[Dict[str, Dict[str, Any]]] = None
) -> EngineManager:
    """Create an optimized engine manager."""
    if service_container is None:
        service_container = ServiceContainer()
    
    manager = EngineManager(service_container)
    
    # Register available engines
    await _register_available_engines(manager.registry)
    
    # Use custom configurations or defaults
    configs = custom_configs or create_default_engine_configs()
    
    # Initialize engines
    success = await manager.initialize_engines(configs)
    if not success:
        raise RuntimeError("Failed to initialize engine manager")
    
    return manager

async def _register_available_engines(registry: EngineRegistry) -> None:
    """Register available engines in the registry."""
    # Specific implementations are registered in their modules
    # This is the centralized registration point
    logger.info("🔧 Registering available engines...")

# =============================================================================
# 📊 ENGINE UTILITIES
# =============================================================================

def validate_engine_config(engine_type: str, config: Dict[str, Any]) -> bool:
    """Validate engine configuration."""
    required_fields = {
        'speed': ['enable_uvloop', 'cache_size'],
        'nlp': ['primary_llm', 'embedding_model'],
        'langchain': ['llm_provider', 'llm_model'],
        'evolution': ['optimization_strategy', 'learning_mode']
    }
    
    if engine_type in required_fields:
        for field in required_fields[engine_type]:
            if field not in config:
                logger.error(f"Missing required field '{field}' for engine '{engine_type}'")
                return False
    
    return True

def merge_engine_configs(
    base_config: Dict[str, Any], 
    override_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge engine configurations with override priority."""
    merged = base_config.copy()
    merged.update(override_config)
    return merged

def create_engine_config(
    engine_type: str,
    **kwargs
) -> Dict[str, Any]:
    """Create a basic engine configuration."""
    base_config = {
        'enabled': True,
        'max_workers': 4,
        'timeout': 30.0,
        'retry_attempts': 3,
        'cache_size': 1000
    }
    
    # Add engine-specific defaults
    if engine_type == 'speed':
        base_config.update({
            'enable_uvloop': True,
            'enable_fast_cache': True
        })
    elif engine_type == 'nlp':
        base_config.update({
            'primary_llm': 'gpt-4-turbo-preview',
            'embedding_model': 'text-embedding-3-large'
        })
    
    # Override with provided kwargs
    base_config.update(kwargs)
    return base_config

# =============================================================================
# 🌟 EXPORTS
# =============================================================================

__all__ = [
    # Core classes
    "EngineRegistry",
    "EngineManager",
    "EngineConfig",
    
    # Factory functions
    "create_default_engine_configs",
    "create_optimized_engine_manager",
    
    # Utility functions
    "validate_engine_config",
    "merge_engine_configs",
    "create_engine_config"
] 