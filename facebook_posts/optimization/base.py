from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import time
import json
from typing import Any, List, Dict, Optional
import logging
"""
🔧 Optimization Base - Sistema Base de Optimización
=================================================

Sistema base para optimizadores con patrón Strategy y Factory.
Proporciona interfaces y clases base para todos los optimizadores.
"""


# ===== BASE CLASSES =====

class Optimizer(ABC):
    """Clase base abstracta para todos los optimizadores."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
self.name = name
        self.config = config or {}
        self.metrics = OptimizationMetrics()
        self.is_enabled = self.config.get('enabled', True)
        self.priority = self.config.get('priority', 1)
    
    @abstractmethod
    async def optimize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimizar datos. Debe ser implementado por subclases."""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Obtener métricas del optimizador."""
        pass
    
    def is_applicable(self, data: Dict[str, Any]) -> bool:
        """Verificar si el optimizador es aplicable a los datos."""
        return self.is_enabled
    
    def get_config(self) -> Dict[str, Any]:
        """Obtener configuración del optimizador."""
        return self.config
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Actualizar configuración del optimizador."""
        self.config.update(new_config)
        self.is_enabled = self.config.get('enabled', True)
        self.priority = self.config.get('priority', 1)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', enabled={self.is_enabled})"

class AsyncOptimizer(Optimizer):
    """Optimizador asíncrono base."""
    
    async def optimize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimización asíncrona base."""
        start_time = time.time()
        
        try:
            if not self.is_applicable(data):
                return data
            
            result = await self._optimize_async(data)
            
            # Actualizar métricas
            processing_time = time.time() - start_time
            self.metrics.record_success(processing_time)
            
            return result
            
        except Exception as e:
            # Actualizar métricas de error
            processing_time = time.time() - start_time
            self.metrics.record_error(processing_time, str(e))
            raise
    
    @abstractmethod
    async def _optimize_async(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Implementación asíncrona de optimización."""
        pass

class SyncOptimizer(Optimizer):
    """Optimizador síncrono base."""
    
    async def optimize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimización síncrona ejecutada de forma asíncrona."""
        start_time = time.time()
        
        try:
            if not self.is_applicable(data):
                return data
            
            # Ejecutar en thread pool para no bloquear
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._optimize_sync, data)
            
            # Actualizar métricas
            processing_time = time.time() - start_time
            self.metrics.record_success(processing_time)
            
            return result
            
        except Exception as e:
            # Actualizar métricas de error
            processing_time = time.time() - start_time
            self.metrics.record_error(processing_time, str(e))
            raise
    
    @abstractmethod
    def _optimize_sync(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Implementación síncrona de optimización."""
        pass

# ===== METRICS AND MONITORING =====

@dataclass
class OptimizationMetrics:
    """Métricas de optimización."""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_processing_time: float = 0.0
    avg_processing_time: float = 0.0
    min_processing_time: float = float('inf')
    max_processing_time: float = 0.0
    last_operation_time: Optional[datetime] = None
    error_count: int = 0
    errors: List[str] = field(default_factory=list)
    
    def record_success(self, processing_time: float) -> None:
        """Registrar operación exitosa."""
        self.total_operations += 1
        self.successful_operations += 1
        self.total_processing_time += processing_time
        self.avg_processing_time = self.total_processing_time / self.total_operations
        self.min_processing_time = min(self.min_processing_time, processing_time)
        self.max_processing_time = max(self.max_processing_time, processing_time)
        self.last_operation_time = datetime.now()
    
    def record_error(self, processing_time: float, error_message: str) -> None:
        """Registrar operación fallida."""
        self.total_operations += 1
        self.failed_operations += 1
        self.total_processing_time += processing_time
        self.avg_processing_time = self.total_processing_time / self.total_operations
        self.min_processing_time = min(self.min_processing_time, processing_time)
        self.max_processing_time = max(self.max_processing_time, processing_time)
        self.last_operation_time = datetime.now()
        self.error_count += 1
        self.errors.append(error_message)
        
        # Mantener solo los últimos 10 errores
        if len(self.errors) > 10:
            self.errors = self.errors[-10:]
    
    def get_success_rate(self) -> float:
        """Obtener tasa de éxito."""
        if self.total_operations == 0:
            return 0.0
        return self.successful_operations / self.total_operations
    
    def get_error_rate(self) -> float:
        """Obtener tasa de error."""
        if self.total_operations == 0:
            return 0.0
        return self.failed_operations / self.total_operations
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            'total_operations': self.total_operations,
            'successful_operations': self.successful_operations,
            'failed_operations': self.failed_operations,
            'total_processing_time': self.total_processing_time,
            'avg_processing_time': self.avg_processing_time,
            'min_processing_time': self.min_processing_time if self.min_processing_time != float('inf') else 0.0,
            'max_processing_time': self.max_processing_time,
            'last_operation_time': self.last_operation_time.isoformat() if self.last_operation_time else None,
            'success_rate': self.get_success_rate(),
            'error_rate': self.get_error_rate(),
            'error_count': self.error_count,
            'recent_errors': self.errors[-5:]  # Últimos 5 errores
        }
    
    def reset(self) -> None:
        """Resetear métricas."""
        self.__init__()

# ===== OPTIMIZATION CONTEXT =====

@dataclass
class OptimizationContext:
    """Contexto de optimización."""
    request_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            'request_id': self.request_id,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }

@dataclass
class OptimizationResult:
    """Resultado de optimización."""
    success: bool
    original_data: Dict[str, Any]
    optimized_data: Dict[str, Any]
    context: OptimizationContext
    processing_time: float
    optimizations_applied: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            'success': self.success,
            'original_data': self.original_data,
            'optimized_data': self.optimized_data,
            'context': self.context.to_dict(),
            'processing_time': self.processing_time,
            'optimizations_applied': self.optimizations_applied,
            'metrics': self.metrics,
            'error': self.error
        }

# ===== OPTIMIZATION PIPELINE =====

class OptimizationPipeline:
    """Pipeline de optimización que ejecuta múltiples optimizadores."""
    
    def __init__(self, optimizers: Optional[List[Optimizer]] = None):
        
    """__init__ function."""
self.optimizers = optimizers or []
        self.context: Optional[OptimizationContext] = None
    
    def add_optimizer(self, optimizer: Optimizer) -> None:
        """Añadir optimizador al pipeline."""
        self.optimizers.append(optimizer)
    
    def remove_optimizer(self, name: str) -> None:
        """Remover optimizador del pipeline."""
        self.optimizers = [opt for opt in self.optimizers if opt.name != name]
    
    def get_optimizer(self, name: str) -> Optional[Optimizer]:
        """Obtener optimizador por nombre."""
        for optimizer in self.optimizers:
            if optimizer.name == name:
                return optimizer
        return None
    
    def sort_optimizers(self) -> None:
        """Ordenar optimizadores por prioridad."""
        self.optimizers.sort(key=lambda opt: opt.priority, reverse=True)
    
    async def optimize(self, data: Dict[str, Any], context: Optional[OptimizationContext] = None) -> OptimizationResult:
        """Ejecutar pipeline de optimización."""
        start_time = time.time()
        self.context = context or OptimizationContext(request_id=str(time.time()))
        
        original_data = data.copy()
        optimized_data = data
        optimizations_applied = []
        metrics = {}
        
        try:
            # Ordenar optimizadores por prioridad
            self.sort_optimizers()
            
            # Ejecutar optimizadores
            for optimizer in self.optimizers:
                if optimizer.is_applicable(optimized_data):
                    try:
                        result = await optimizer.optimize(optimized_data)
                        optimized_data = result
                        optimizations_applied.append(optimizer.name)
                        metrics[optimizer.name] = optimizer.get_metrics()
                    except Exception as e:
                        # Continuar con el siguiente optimizador si uno falla
                        print(f"Optimizer {optimizer.name} failed: {e}")
                        continue
            
            processing_time = time.time() - start_time
            
            return OptimizationResult(
                success=True,
                original_data=original_data,
                optimized_data=optimized_data,
                context=self.context,
                processing_time=processing_time,
                optimizations_applied=optimizations_applied,
                metrics=metrics
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            return OptimizationResult(
                success=False,
                original_data=original_data,
                optimized_data=optimized_data,
                context=self.context,
                processing_time=processing_time,
                optimizations_applied=optimizations_applied,
                metrics=metrics,
                error=str(e)
            )
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Obtener métricas del pipeline completo."""
        pipeline_metrics = {
            'total_optimizers': len(self.optimizers),
            'enabled_optimizers': len([opt for opt in self.optimizers if opt.is_enabled]),
            'optimizers': {}
        }
        
        for optimizer in self.optimizers:
            pipeline_metrics['optimizers'][optimizer.name] = {
                'enabled': optimizer.is_enabled,
                'priority': optimizer.priority,
                'metrics': optimizer.get_metrics()
            }
        
        return pipeline_metrics

# ===== OPTIMIZATION FACTORY =====

class OptimizerFactory:
    """Factory para crear optimizadores."""
    
    _optimizers: Dict[str, Type[Optimizer]] = {}
    _default_configs: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(
        cls, 
        name: str, 
        optimizer_class: Type[Optimizer], 
        default_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Registrar un optimizador."""
        cls._optimizers[name] = optimizer_class
        if default_config:
            cls._default_configs[name] = default_config
    
    @classmethod
    def create(cls, name: str, config: Optional[Dict[str, Any]] = None, **kwargs) -> Optimizer:
        """Crear un optimizador."""
        if name not in cls._optimizers:
            raise ValueError(f"Optimizer '{name}' not found. Available: {list(cls._optimizers.keys())}")
        
        optimizer_class = cls._optimizers[name]
        default_config = cls._default_configs.get(name, {})
        
        # Combinar configuraciones
        final_config = default_config.copy()
        if config:
            final_config.update(config)
        final_config.update(kwargs)
        
        return optimizer_class(name, final_config)
    
    @classmethod
    def get_available(cls) -> List[str]:
        """Obtener optimizadores disponibles."""
        return list(cls._optimizers.keys())
    
    @classmethod
    def get_optimizer_class(cls, name: str) -> Optional[Type[Optimizer]]:
        """Obtener clase de optimizador."""
        return cls._optimizers.get(name)
    
    @classmethod
    def unregister(cls, name: str) -> None:
        """Desregistrar un optimizador."""
        if name in cls._optimizers:
            del cls._optimizers[name]
        if name in cls._default_configs:
            del cls._default_configs[name]

# ===== UTILITY DECORATORS =====

def optimizer(name: str, default_config: Optional[Dict[str, Any]] = None):
    """Decorador para registrar optimizadores automáticamente."""
    def decorator(cls: Type[Optimizer]) -> Type[Optimizer]:
        OptimizerFactory.register(name, cls, default_config)
        return cls
    return decorator

def require_config(*required_keys: str):
    """Decorador para validar configuración requerida."""
    def decorator(func) -> Any:
        def wrapper(self, data: Dict[str, Any]) -> Dict[str, Any]:
            missing_keys = [key for key in required_keys if key not in self.config]
            if missing_keys:
                raise ValueError(f"Missing required config keys: {missing_keys}")
            return func(self, data)
        return wrapper
    return decorator

# ===== EXPORTS =====

__all__ = [
    # Base Classes
    'Optimizer',
    'AsyncOptimizer',
    'SyncOptimizer',
    
    # Metrics
    'OptimizationMetrics',
    
    # Context and Results
    'OptimizationContext',
    'OptimizationResult',
    
    # Pipeline
    'OptimizationPipeline',
    
    # Factory
    'OptimizerFactory',
    
    # Decorators
    'optimizer',
    'require_config'
] 