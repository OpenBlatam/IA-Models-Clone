from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import time
from ..core.interfaces import (
from typing import Any, List, Dict, Optional
"""
🚀 BLATAM AI ENGINE MANAGER v5.0.0
==================================

Gestor modular de motores AI:
- 🏭 Engine registry y discovery
- 🔧 Dependency resolution
- ⚡ Parallel initialization
- 📊 Health monitoring
- 🎯 Auto-routing de operaciones
"""


    AIEngine, ComponentStatus, ProcessingType, OptimizationStrategy,
    EngineFactory, EventPublisher, BlatamComponent
)

logger = logging.getLogger(__name__)

# =============================================================================
# 📊 ENGINE METADATA
# =============================================================================

@dataclass
class EngineMetadata:
    """Metadata de un motor."""
    name: str
    engine_type: str
    dependencies: List[str] = field(default_factory=list)
    supported_types: List[ProcessingType] = field(default_factory=list)
    priority: int = 1
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    capabilities: Set[str] = field(default_factory=set)

@dataclass
class EngineStatus:
    """Estado de un motor."""
    status: ComponentStatus
    last_health_check: datetime = field(default_factory=datetime.now)
    error_count: int = 0
    avg_response_time_ms: float = 0.0
    total_requests: int = 0
    success_rate: float = 100.0

# =============================================================================
# 🏗️ ENGINE REGISTRY
# =============================================================================

class EngineRegistry:
    """Registro modular de motores disponibles."""
    
    def __init__(self) -> Any:
        self._factories: Dict[str, EngineFactory] = {}
        self._metadata: Dict[str, EngineMetadata] = {}
        self._default_configs: Dict[str, Dict[str, Any]] = {}
    
    def register_engine(
        self,
        engine_type: str,
        factory: EngineFactory,
        metadata: EngineMetadata,
        default_config: Optional[Dict[str, Any]] = None
    ):
        """Registra un motor en el registry."""
        self._factories[engine_type] = factory
        self._metadata[engine_type] = metadata
        if default_config:
            self._default_configs[engine_type] = default_config
        
        logger.info(f"🔧 Registered engine: {engine_type} with capabilities: {metadata.capabilities}")
    
    def get_available_engines(self) -> List[str]:
        """Obtiene motores disponibles."""
        return list(self._factories.keys())
    
    def get_engine_metadata(self, engine_type: str) -> Optional[EngineMetadata]:
        """Obtiene metadata de un motor."""
        return self._metadata.get(engine_type)
    
    def get_factory(self, engine_type: str) -> Optional[EngineFactory]:
        """Obtiene factory de un motor."""
        return self._factories.get(engine_type)
    
    def get_default_config(self, engine_type: str) -> Dict[str, Any]:
        """Obtiene configuración por defecto."""
        return self._default_configs.get(engine_type, {})
    
    def resolve_dependencies(self, engine_types: List[str]) -> List[str]:
        """Resuelve orden de dependencias."""
        resolved = []
        remaining = set(engine_types)
        
        while remaining:
            ready = []
            
            for engine_type in remaining:
                metadata = self._metadata.get(engine_type)
                if not metadata:
                    continue
                
                # Check if all dependencies are resolved
                if all(dep in resolved for dep in metadata.dependencies):
                    ready.append(engine_type)
            
            if not ready:
                raise ValueError(f"Circular dependency detected: {remaining}")
            
            # Sort by priority
            ready.sort(key=lambda x: self._metadata[x].priority, reverse=True)
            
            for engine_type in ready:
                resolved.append(engine_type)
                remaining.remove(engine_type)
        
        return resolved
    
    def find_engines_for_processing(self, processing_type: ProcessingType) -> List[str]:
        """Encuentra motores que pueden manejar un tipo de procesamiento."""
        suitable_engines = []
        
        for engine_type, metadata in self._metadata.items():
            if processing_type in metadata.supported_types:
                suitable_engines.append(engine_type)
        
        # Sort by priority
        suitable_engines.sort(
            key=lambda x: self._metadata[x].priority,
            reverse=True
        )
        
        return suitable_engines

# =============================================================================
# 🎯 ENGINE MANAGER
# =============================================================================

class ModularEngineManager:
    """Gestor modular de motores AI."""
    
    def __init__(self, event_publisher: Optional[EventPublisher] = None):
        
    """__init__ function."""
self.registry = EngineRegistry()
        self.engines: Dict[str, AIEngine] = {}
        self.engine_status: Dict[str, EngineStatus] = {}
        self.event_publisher = event_publisher
        self.is_initialized = False
        self.initialization_order: List[str] = []
        
        # Performance tracking
        self.operation_stats: Dict[str, List[float]] = {}
        self.routing_cache: Dict[str, str] = {}
    
    async def initialize_engines(
        self,
        engine_configs: Dict[str, Dict[str, Any]],
        enabled_engines: Optional[List[str]] = None,
        parallel_init: bool = True
    ) -> bool:
        """Inicializa motores de forma modular."""
        try:
            logger.info("🚀 Starting modular engine initialization...")
            start_time = time.time()
            
            # Determine engines to initialize
            if enabled_engines is None:
                enabled_engines = list(engine_configs.keys())
            
            # Validate all engines are registered
            for engine_type in enabled_engines:
                if engine_type not in self.registry.get_available_engines():
                    raise ValueError(f"Engine '{engine_type}' not registered")
            
            # Resolve dependencies
            self.initialization_order = self.registry.resolve_dependencies(enabled_engines)
            logger.info(f"🔧 Initialization order: {self.initialization_order}")
            
            # Initialize engines
            if parallel_init:
                success = await self._parallel_initialization(engine_configs)
            else:
                success = await self._sequential_initialization(engine_configs)
            
            if success:
                self.is_initialized = True
                init_time = (time.time() - start_time) * 1000
                logger.info(f"✅ Engines initialized in {init_time:.2f}ms")
                
                # Start health monitoring
                asyncio.create_task(self._health_monitoring_loop())
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Engine initialization failed: {e}")
            return False
    
    async def _parallel_initialization(self, engine_configs: Dict[str, Dict[str, Any]]) -> bool:
        """Inicialización paralela respetando dependencias."""
        initialized = set()
        
        # Group by dependency levels
        dependency_levels = self._group_by_dependency_level()
        
        for level_engines in dependency_levels:
            # Initialize all engines in this level in parallel
            tasks = []
            for engine_type in level_engines:
                if engine_type in engine_configs:
                    task = self._initialize_single_engine(engine_type, engine_configs[engine_type])
                    tasks.append((engine_type, task))
            
            # Wait for all engines in this level
            results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            for (engine_type, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    logger.error(f"❌ Failed to initialize {engine_type}: {result}")
                    return False
                elif result:
                    initialized.add(engine_type)
                    logger.info(f"✅ Engine '{engine_type}' initialized")
                else:
                    logger.error(f"❌ Engine '{engine_type}' initialization failed")
                    return False
        
        return True
    
    async def _sequential_initialization(self, engine_configs: Dict[str, Dict[str, Any]]) -> bool:
        """Inicialización secuencial."""
        for engine_type in self.initialization_order:
            if engine_type in engine_configs:
                success = await self._initialize_single_engine(engine_type, engine_configs[engine_type])
                if not success:
                    logger.error(f"❌ Failed to initialize engine: {engine_type}")
                    return False
                logger.info(f"✅ Engine '{engine_type}' initialized")
        
        return True
    
    async def _initialize_single_engine(self, engine_type: str, config: Dict[str, Any]) -> bool:
        """Inicializa un motor específico."""
        try:
            # Get factory and create engine
            factory = self.registry.get_factory(engine_type)
            if not factory:
                raise ValueError(f"No factory found for engine: {engine_type}")
            
            # Merge with default config
            default_config = self.registry.get_default_config(engine_type)
            merged_config = {**default_config, **config}
            
            # Validate config
            if not factory.validate_config(merged_config):
                raise ValueError(f"Invalid config for engine: {engine_type}")
            
            # Create engine
            engine = await factory.create(merged_config)
            
            # Initialize engine
            init_success = await engine.initialize()
            if not init_success:
                raise RuntimeError(f"Engine initialization failed: {engine_type}")
            
            # Store engine and status
            self.engines[engine_type] = engine
            self.engine_status[engine_type] = EngineStatus(status=ComponentStatus.READY)
            
            # Publish event
            if self.event_publisher:
                await self.event_publisher.publish("engine_initialized", {
                    "engine_type": engine_type,
                    "status": "success"
                })
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error initializing engine '{engine_type}': {e}")
            
            # Publish failure event
            if self.event_publisher:
                await self.event_publisher.publish("engine_initialization_failed", {
                    "engine_type": engine_type,
                    "error": str(e)
                })
            
            return False
    
    def _group_by_dependency_level(self) -> List[List[str]]:
        """Agrupa motores por nivel de dependencia."""
        levels = []
        remaining = set(self.initialization_order)
        
        while remaining:
            current_level = []
            
            for engine_type in list(remaining):
                metadata = self.registry.get_engine_metadata(engine_type)
                if not metadata:
                    continue
                
                # Check if all dependencies are in previous levels
                deps_satisfied = all(
                    dep not in remaining for dep in metadata.dependencies
                )
                
                if deps_satisfied:
                    current_level.append(engine_type)
            
            if not current_level:
                break
            
            for engine_type in current_level:
                remaining.remove(engine_type)
            
            levels.append(current_level)
        
        return levels
    
    # =========================================================================
    # 🎯 ENGINE ACCESS & ROUTING
    # =========================================================================
    
    def get_engine(self, engine_type: str) -> Optional[AIEngine]:
        """Obtiene un motor específico."""
        return self.engines.get(engine_type)
    
    def get_all_engines(self) -> Dict[str, AIEngine]:
        """Obtiene todos los motores."""
        return self.engines.copy()
    
    def get_ready_engines(self) -> Dict[str, AIEngine]:
        """Obtiene motores listos."""
        ready_engines = {}
        for engine_type, engine in self.engines.items():
            status = self.engine_status.get(engine_type)
            if status and status.status in [ComponentStatus.READY, ComponentStatus.RUNNING]:
                ready_engines[engine_type] = engine
        return ready_engines
    
    async def route_processing(
        self,
        processing_type: ProcessingType,
        data: Any,
        **kwargs
    ) -> Tuple[str, Any]:
        """Enruta procesamiento al motor más apropiado."""
        # Check cache first
        cache_key = f"{processing_type.value}_{hash(str(type(data)))}"
        if cache_key in self.routing_cache:
            preferred_engine = self.routing_cache[cache_key]
            if preferred_engine in self.engines:
                engine = self.engines[preferred_engine]
                result = await engine.process(data, **kwargs)
                return preferred_engine, result
        
        # Find suitable engines
        suitable_engines = self.registry.find_engines_for_processing(processing_type)
        ready_engines = self.get_ready_engines()
        
        # Filter to available and ready engines
        available_engines = [
            engine_type for engine_type in suitable_engines
            if engine_type in ready_engines
        ]
        
        if not available_engines:
            raise RuntimeError(f"No engines available for processing type: {processing_type}")
        
        # Select best engine (for now, use first available)
        selected_engine = available_engines[0]
        
        # Cache the routing decision
        self.routing_cache[cache_key] = selected_engine
        
        # Process with selected engine
        engine = self.engines[selected_engine]
        result = await engine.process(data, **kwargs)
        
        return selected_engine, result
    
    async def auto_route(self, data: Any, **kwargs) -> Tuple[str, Any]:
        """Auto-enruta basado en el tipo de datos."""
        processing_type = self._detect_processing_type(data)
        return await self.route_processing(processing_type, data, **kwargs)
    
    def _detect_processing_type(self, data: Any) -> ProcessingType:
        """Detecta tipo de procesamiento basado en datos."""
        if isinstance(data, str):
            if len(data) < 100:
                return ProcessingType.NLP_ANALYSIS
            else:
                return ProcessingType.NLP_GENERATION
        elif isinstance(data, dict):
            if 'product_name' in data or 'features' in data:
                return ProcessingType.PRODUCT_DESCRIPTION
            else:
                return ProcessingType.ENTERPRISE
        elif isinstance(data, (list, tuple)):
            return ProcessingType.BATCH_PROCESSING
        else:
            return ProcessingType.ENTERPRISE
    
    # =========================================================================
    # 📊 MONITORING & HEALTH
    # =========================================================================
    
    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Health check de todos los motores."""
        health_results = {}
        
        tasks = []
        for engine_type, engine in self.engines.items():
            task = self._health_check_single(engine_type, engine)
            tasks.append((engine_type, task))
        
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        for (engine_type, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                health_results[engine_type] = {
                    'status': 'error',
                    'error': str(result)
                }
            else:
                health_results[engine_type] = result
        
        return health_results
    
    async def _health_check_single(self, engine_type: str, engine: AIEngine) -> Dict[str, Any]:
        """Health check de un motor específico."""
        try:
            start_time = time.time()
            health_info = await engine.health_check()
            response_time = (time.time() - start_time) * 1000
            
            # Update status
            if engine_type in self.engine_status:
                status = self.engine_status[engine_type]
                status.last_health_check = datetime.now()
                status.avg_response_time_ms = (
                    status.avg_response_time_ms * 0.9 + response_time * 0.1
                )
            
            return {
                **health_info,
                'response_time_ms': response_time
            }
            
        except Exception as e:
            # Update error count
            if engine_type in self.engine_status:
                self.engine_status[engine_type].error_count += 1
                self.engine_status[engine_type].status = ComponentStatus.ERROR
            
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def _health_monitoring_loop(self) -> Any:
        """Loop de monitoreo de salud."""
        while self.is_initialized:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self.health_check_all()
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
    
    def get_engine_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Obtiene estadísticas de motores."""
        stats = {}
        
        for engine_type, engine in self.engines.items():
            try:
                engine_stats = engine.get_stats()
                status = self.engine_status.get(engine_type)
                
                stats[engine_type] = {
                    **engine_stats,
                    'status': status.status.value if status else 'unknown',
                    'error_count': status.error_count if status else 0,
                    'avg_response_time_ms': status.avg_response_time_ms if status else 0,
                    'success_rate': status.success_rate if status else 0
                }
                
            except Exception as e:
                stats[engine_type] = {'error': str(e)}
        
        return stats
    
    # =========================================================================
    # 🔄 OPTIMIZATION
    # =========================================================================
    
    async def optimize_all_engines(self, strategy: OptimizationStrategy = OptimizationStrategy.BALANCED) -> Dict[str, Dict[str, Any]]:
        """Optimiza todos los motores."""
        optimization_results = {}
        
        for engine_type, engine in self.engines.items():
            try:
                if hasattr(engine, 'optimize'):
                    metrics = self._get_engine_metrics(engine_type)
                    result = await engine.optimize(strategy, metrics)
                    optimization_results[engine_type] = result
                else:
                    optimization_results[engine_type] = {'status': 'not_optimizable'}
                    
            except Exception as e:
                optimization_results[engine_type] = {'error': str(e)}
        
        return optimization_results
    
    def _get_engine_metrics(self, engine_type: str) -> Dict[str, Any]:
        """Obtiene métricas de un motor para optimización."""
        status = self.engine_status.get(engine_type)
        if not status:
            return {}
        
        return {
            'avg_response_time_ms': status.avg_response_time_ms,
            'error_count': status.error_count,
            'success_rate': status.success_rate,
            'total_requests': status.total_requests
        }
    
    # =========================================================================
    # 🛠️ UTILITIES
    # =========================================================================
    
    async def shutdown_all(self) -> bool:
        """Apaga todos los motores."""
        logger.info("🛑 Shutting down all engines...")
        
        # Shutdown in reverse order
        shutdown_order = list(reversed(self.initialization_order))
        
        for engine_type in shutdown_order:
            if engine_type in self.engines:
                try:
                    engine = self.engines[engine_type]
                    await engine.shutdown()
                    logger.info(f"✅ Engine '{engine_type}' shutdown")
                except Exception as e:
                    logger.error(f"❌ Error shutting down '{engine_type}': {e}")
        
        self.engines.clear()
        self.engine_status.clear()
        self.is_initialized = False
        
        return True

# =============================================================================
# 🌟 EXPORTS
# =============================================================================

__all__ = [
    "EngineMetadata",
    "EngineStatus", 
    "EngineRegistry",
    "ModularEngineManager"
] 