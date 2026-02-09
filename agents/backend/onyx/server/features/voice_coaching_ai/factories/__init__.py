"""
🏭 VOICE COACHING AI - FACTORY MODULE
====================================

Factory patterns for creating voice coaching components with proper
dependency injection, configuration management, and component lifecycle.
"""

import logging
from typing import Dict, Optional, Any, Type
from dataclasses import dataclass

from ..core import (
    VoiceCoachingConfig, VoiceCoachingComponent, 
    create_default_voice_config, PerformanceMetrics
)

logger = logging.getLogger(__name__)

# =============================================================================
# 🏭 FACTORY CONFIGURATIONS
# =============================================================================

@dataclass
class FactoryConfig:
    """Configuration for factory operations"""
    enable_caching: bool = True
    cache_ttl: int = 300  # 5 minutes
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_metrics: bool = True
    enable_logging: bool = True

class ComponentRegistry:
    """Registry for voice coaching components"""
    
    def __init__(self):
        self._components: Dict[str, Any] = {}
        self._factories: Dict[str, Type] = {}
        self._configs: Dict[str, VoiceCoachingConfig] = {}
    
    def register_component(self, name: str, component: Any, config: VoiceCoachingConfig):
        """Register a component instance"""
        self._components[name] = component
        self._configs[name] = config
        logger.info(f"Registered component: {name}")
    
    def register_factory(self, name: str, factory_class: Type):
        """Register a factory class"""
        self._factories[name] = factory_class
        logger.info(f"Registered factory: {name}")
    
    def get_component(self, name: str) -> Optional[Any]:
        """Get a registered component"""
        return self._components.get(name)
    
    def get_factory(self, name: str) -> Optional[Type]:
        """Get a registered factory"""
        return self._factories.get(name)
    
    def get_config(self, name: str) -> Optional[VoiceCoachingConfig]:
        """Get component configuration"""
        return self._configs.get(name)
    
    def list_components(self) -> Dict[str, str]:
        """List all registered components"""
        return {name: type(comp).__name__ for name, comp in self._components.items()}

# =============================================================================
# 🏭 BASE FACTORY CLASSES
# =============================================================================

class VoiceCoachingFactory:
    """Base factory for voice coaching components"""
    
    def __init__(self, config: VoiceCoachingConfig, factory_config: FactoryConfig = None):
        self.config = config
        self.factory_config = factory_config or FactoryConfig()
        self.registry = ComponentRegistry()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def create_component(self, component_type: str, **kwargs) -> Optional[VoiceCoachingComponent]:
        """Create a component of the specified type"""
        try:
            if component_type == "engine":
                return await self._create_engine(**kwargs)
            elif component_type == "service":
                return await self._create_service(**kwargs)
            elif component_type == "analyzer":
                return await self._create_analyzer(**kwargs)
            elif component_type == "coach":
                return await self._create_coach(**kwargs)
            else:
                self.logger.error(f"Unknown component type: {component_type}")
                return None
        except Exception as e:
            self.logger.error(f"Failed to create component {component_type}: {e}")
            return None
    
    async def _create_engine(self, **kwargs) -> Optional[VoiceCoachingComponent]:
        """Create voice coaching engine"""
        try:
            from ..engines.openrouter_voice_engine import OpenRouterVoiceEngine
            engine = OpenRouterVoiceEngine(self.config)
            await engine.initialize()
            return engine
        except Exception as e:
            self.logger.error(f"Failed to create engine: {e}")
            return None
    
    async def _create_service(self, **kwargs) -> Optional[VoiceCoachingComponent]:
        """Create voice coaching service"""
        try:
            from ..services.voice_coaching_service import VoiceCoachingService
            service = VoiceCoachingService(self.config)
            await service.initialize()
            return service
        except Exception as e:
            self.logger.error(f"Failed to create service: {e}")
            return None
    
    async def _create_analyzer(self, **kwargs) -> Optional[VoiceCoachingComponent]:
        """Create voice analyzer"""
        # Placeholder for future analyzer implementations
        self.logger.warning("Voice analyzer creation not implemented yet")
        return None
    
    async def _create_coach(self, **kwargs) -> Optional[VoiceCoachingComponent]:
        """Create voice coach"""
        # Placeholder for future coach implementations
        self.logger.warning("Voice coach creation not implemented yet")
        return None

# =============================================================================
# 🏭 SPECIALIZED FACTORIES
# =============================================================================

class EngineFactory(VoiceCoachingFactory):
    """Factory for creating voice coaching engines"""
    
    async def create_openrouter_engine(self, api_key: str, model: str = "openai/gpt-4-turbo") -> Optional[VoiceCoachingComponent]:
        """Create OpenRouter voice coaching engine"""
        try:
            config = VoiceCoachingConfig(
                openrouter_api_key=api_key,
                openrouter_model=model,
                enable_voice_analysis=True,
                enable_confidence_measurement=True,
                enable_leadership_coaching=True
            )
            
            from ..engines.openrouter_voice_engine import OpenRouterVoiceEngine
            engine = OpenRouterVoiceEngine(config)
            
            if await engine.initialize():
                self.registry.register_component("openrouter_engine", engine, config)
                return engine
            else:
                self.logger.error("Failed to initialize OpenRouter engine")
                return None
        except Exception as e:
            self.logger.error(f"Failed to create OpenRouter engine: {e}")
            return None

class ServiceFactory(VoiceCoachingFactory):
    """Factory for creating voice coaching services"""
    
    async def create_coaching_service(self, engine: VoiceCoachingComponent) -> Optional[VoiceCoachingComponent]:
        """Create voice coaching service with engine dependency"""
        try:
            from ..services.voice_coaching_service import VoiceCoachingService
            service = VoiceCoachingService(self.config)
            
            # Inject engine dependency
            if hasattr(service, 'engine'):
                service.engine = engine
            
            if await service.initialize():
                self.registry.register_component("coaching_service", service, self.config)
                return service
            else:
                self.logger.error("Failed to initialize coaching service")
                return None
        except Exception as e:
            self.logger.error(f"Failed to create coaching service: {e}")
            return None

# =============================================================================
# 🏭 FACTORY MANAGER
# =============================================================================

class VoiceCoachingFactoryManager:
    """Manager for voice coaching factories"""
    
    def __init__(self):
        self.factories: Dict[str, VoiceCoachingFactory] = {}
        self.registry = ComponentRegistry()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def register_factory(self, name: str, factory: VoiceCoachingFactory):
        """Register a factory"""
        self.factories[name] = factory
        self.logger.info(f"Registered factory: {name}")
    
    def get_factory(self, name: str) -> Optional[VoiceCoachingFactory]:
        """Get a factory by name"""
        return self.factories.get(name)
    
    async def create_complete_system(self, api_key: str, model: str = "openai/gpt-4-turbo") -> Dict[str, Any]:
        """Create a complete voice coaching system"""
        try:
            # Create configuration
            config = create_default_voice_config()
            config.openrouter_api_key = api_key
            config.openrouter_model = model
            
            # Create engine factory
            engine_factory = EngineFactory(config)
            engine = await engine_factory.create_openrouter_engine(api_key, model)
            
            if not engine:
                raise ValueError("Failed to create voice coaching engine")
            
            # Create service factory
            service_factory = ServiceFactory(config)
            service = await service_factory.create_coaching_service(engine)
            
            if not service:
                raise ValueError("Failed to create voice coaching service")
            
            # Register components
            self.registry.register_component("engine", engine, config)
            self.registry.register_component("service", service, config)
            
            return {
                "engine": engine,
                "service": service,
                "config": config,
                "status": "initialized"
            }
        except Exception as e:
            self.logger.error(f"Failed to create complete system: {e}")
            return {"status": "failed", "error": str(e)}

# =============================================================================
# 🚀 FACTORY FUNCTIONS
# =============================================================================

def create_voice_coaching_factory(config: VoiceCoachingConfig = None) -> VoiceCoachingFactory:
    """Create a voice coaching factory"""
    if config is None:
        config = create_default_voice_config()
    return VoiceCoachingFactory(config)

def create_engine_factory(config: VoiceCoachingConfig = None) -> EngineFactory:
    """Create an engine factory"""
    if config is None:
        config = create_default_voice_config()
    return EngineFactory(config)

def create_service_factory(config: VoiceCoachingConfig = None) -> ServiceFactory:
    """Create a service factory"""
    if config is None:
        config = create_default_voice_config()
    return ServiceFactory(config)

def create_factory_manager() -> VoiceCoachingFactoryManager:
    """Create a factory manager"""
    return VoiceCoachingFactoryManager()

# =============================================================================
# 🎯 QUICK FACTORY FUNCTIONS
# =============================================================================

async def create_voice_coaching_system(api_key: str, model: str = "openai/gpt-4-turbo") -> Dict[str, Any]:
    """Quick function to create a complete voice coaching system"""
    manager = create_factory_manager()
    return await manager.create_complete_system(api_key, model)

async def create_voice_engine(api_key: str, model: str = "openai/gpt-4-turbo"):
    """Quick function to create a voice coaching engine"""
    factory = create_engine_factory()
    return await factory.create_openrouter_engine(api_key, model) 