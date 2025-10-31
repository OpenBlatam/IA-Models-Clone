from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from typing import Dict, Any, Optional
from enum import Enum
from ..core.interfaces.contracts import IOptimizer, ICache, INLPAnalyzer
from ..core.entities.models import OptimizationTier
from ..application.services.nlp_service import NLPAnalysisService
from ..infrastructure.optimization.adapters import UltraOptimizerAdapter, ExtremeOptimizerAdapter
from ..infrastructure.caching.adapters import MemoryCacheAdapter, OptimizedCacheAdapter
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
🏭 FACTORY - Dependency Injection
=================================

Factory modular para crear e inyectar dependencias.
"""




class ComponentType(Enum):
    """Tipos de componentes disponibles."""
    OPTIMIZER = "optimizer"
    CACHE = "cache"
    NLP_SERVICE = "nlp_service"


class ModularFactory:
    """🏭 Factory modular para crear componentes."""
    
    def __init__(self) -> Any:
        self._instances: Dict[str, Any] = {}
        self._config = self._load_default_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Cargar configuración por defecto."""
        return {
            'optimization_tier': OptimizationTier.ULTRA,
            'cache_size': 1000,
            'cache_ttl': 3600,
            'enable_fallback': True,
            'enable_caching': True
        }
    
    def create_optimizer(self, tier: OptimizationTier) -> IOptimizer:
        """Crear optimizador según tier."""
        instance_key = f"optimizer_{tier.value}"
        
        if instance_key not in self._instances:
            if tier == OptimizationTier.EXTREME:
                self._instances[instance_key] = ExtremeOptimizerAdapter()
            elif tier == OptimizationTier.ULTRA:
                self._instances[instance_key] = UltraOptimizerAdapter()
            else:
                # Default to Ultra for standard
                self._instances[instance_key] = UltraOptimizerAdapter()
        
        return self._instances[instance_key]
    
    def create_cache(self, cache_type: str = "optimized") -> ICache:
        """Crear sistema de cache."""
        instance_key = f"cache_{cache_type}"
        
        if instance_key not in self._instances:
            if cache_type == "optimized":
                self._instances[instance_key] = OptimizedCacheAdapter()
            elif cache_type == "memory":
                self._instances[instance_key] = MemoryCacheAdapter(
                    max_size=self._config['cache_size']
                )
            else:
                # Default to memory
                self._instances[instance_key] = MemoryCacheAdapter()
        
        return self._instances[instance_key]
    
    def create_nlp_service(
        self, 
        optimization_tier: OptimizationTier = None,
        cache_type: str = "optimized"
    ) -> INLPAnalyzer:
        """Crear servicio NLP completo."""
        tier = optimization_tier or self._config['optimization_tier']
        instance_key = f"nlp_service_{tier.value}_{cache_type}"
        
        if instance_key not in self._instances:
            optimizer = self.create_optimizer(tier)
            cache = self.create_cache(cache_type)
            
            self._instances[instance_key] = NLPAnalysisService(
                optimizer=optimizer,
                cache=cache
            )
        
        return self._instances[instance_key]
    
    def get_instance(self, component_type: ComponentType, **kwargs) -> Optional[Dict[str, Any]]:
        """Obtener instancia de componente."""
        if component_type == ComponentType.OPTIMIZER:
            tier = kwargs.get('tier', OptimizationTier.ULTRA)
            return self.create_optimizer(tier)
        
        elif component_type == ComponentType.CACHE:
            cache_type = kwargs.get('cache_type', 'optimized')
            return self.create_cache(cache_type)
        
        elif component_type == ComponentType.NLP_SERVICE:
            tier = kwargs.get('optimization_tier', OptimizationTier.ULTRA)
            cache_type = kwargs.get('cache_type', 'optimized')
            return self.create_nlp_service(tier, cache_type)
        
        else:
            raise ValueError(f"Unknown component type: {component_type}")
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configurar factory."""
        self._config.update(config)
        # Clear instances to force recreation with new config
        self._instances.clear()
    
    def clear_instances(self) -> None:
        """Limpiar todas las instancias."""
        self._instances.clear()


# Global factory instance
_factory: Optional[ModularFactory] = None

def get_factory() -> ModularFactory:
    """Obtener factory global."""
    global _factory
    if _factory is None:
        _factory = ModularFactory()
    return _factory


def create_production_nlp_service(
    optimization_tier: OptimizationTier = OptimizationTier.ULTRA
) -> INLPAnalyzer:
    """Crear servicio NLP de producción."""
    factory = get_factory()
    return factory.create_nlp_service(optimization_tier=optimization_tier) 