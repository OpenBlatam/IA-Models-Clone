"""
Middleware Pipeline - Pipeline de Middleware
==========================================

Sistema de pipeline de middleware con ejecución asíncrona y composición dinámica.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime
from enum import Enum

from ..interfaces.base_interfaces import IMiddleware, IComponent

logger = logging.getLogger(__name__)


class MiddlewareType(Enum):
    """Tipos de middleware."""
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    LOGGING = "logging"
    METRICS = "metrics"
    CACHING = "caching"
    RATE_LIMITING = "rate_limiting"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    CUSTOM = "custom"


class MiddlewareContext:
    """Contexto de middleware."""
    
    def __init__(self, request: Any = None, response: Any = None, 
                 metadata: Dict[str, Any] = None):
        self.request = request
        self.response = response
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
        self.modified = False
        self.errors: List[Exception] = []
        self.warnings: List[str] = []
    
    def set_request(self, request: Any) -> None:
        """Establecer request."""
        self.request = request
        self.modified = True
    
    def get_request(self) -> Any:
        """Obtener request."""
        return self.request
    
    def set_response(self, response: Any) -> None:
        """Establecer response."""
        self.response = response
        self.modified = True
    
    def get_response(self) -> Any:
        """Obtener response."""
        return self.response
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Establecer metadatos."""
        self.metadata[key] = value
        self.modified = True
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Obtener metadatos."""
        return self.metadata.get(key, default)
    
    def add_error(self, error: Exception) -> None:
        """Agregar error."""
        self.errors.append(error)
        self.modified = True
    
    def add_warning(self, warning: str) -> None:
        """Agregar advertencia."""
        self.warnings.append(warning)
        self.modified = True
    
    def has_errors(self) -> bool:
        """Verificar si tiene errores."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Verificar si tiene advertencias."""
        return len(self.warnings) > 0


class MiddlewareNode:
    """Nodo de middleware en el pipeline."""
    
    def __init__(self, middleware: IMiddleware, middleware_type: MiddlewareType, 
                 priority: int = 0, enabled: bool = True):
        self.middleware = middleware
        self.middleware_type = middleware_type
        self.priority = priority
        self.enabled = enabled
        self.created_at = datetime.utcnow()
        self.execution_count = 0
        self.last_execution = None
        self.total_execution_time = 0.0
    
    async def execute(self, context: MiddlewareContext) -> MiddlewareContext:
        """Ejecutar middleware."""
        if not self.enabled:
            return context
        
        start_time = datetime.utcnow()
        
        try:
            if self.middleware_type == MiddlewareType.REQUEST:
                processed_request = await self.middleware.process_request(context.request)
                context.set_request(processed_request)
            elif self.middleware_type == MiddlewareType.RESPONSE:
                processed_response = await self.middleware.process_response(context.response)
                context.set_response(processed_response)
            else:
                # Para otros tipos, usar process_request como método genérico
                processed_data = await self.middleware.process_request(context)
                if processed_data:
                    context = processed_data
            
            # Actualizar estadísticas
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self.execution_count += 1
            self.last_execution = datetime.utcnow()
            self.total_execution_time += execution_time
            
            return context
            
        except Exception as e:
            logger.error(f"Error executing middleware {self.middleware.__class__.__name__}: {e}")
            context.add_error(e)
            return context
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del nodo."""
        avg_execution_time = (self.total_execution_time / self.execution_count 
                             if self.execution_count > 0 else 0.0)
        
        return {
            "middleware_name": self.middleware.__class__.__name__,
            "middleware_type": self.middleware_type.value,
            "priority": self.priority,
            "enabled": self.enabled,
            "execution_count": self.execution_count,
            "last_execution": self.last_execution.isoformat() if self.last_execution else None,
            "total_execution_time": self.total_execution_time,
            "average_execution_time": avg_execution_time,
            "created_at": self.created_at.isoformat()
        }


class MiddlewarePipeline(IComponent):
    """Pipeline de middleware."""
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.nodes: List[MiddlewareNode] = []
        self._lock = asyncio.Lock()
        self._initialized = False
        self._execution_stats: Dict[str, Any] = {}
    
    async def initialize(self) -> None:
        """Inicializar pipeline."""
        try:
            self._initialized = True
            logger.info(f"Middleware pipeline {self.name} initialized")
            
        except Exception as e:
            logger.error(f"Error initializing middleware pipeline: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Cerrar pipeline."""
        try:
            # Limpiar nodos
            self.nodes.clear()
            self._initialized = False
            logger.info(f"Middleware pipeline {self.name} shutdown complete")
            
        except Exception as e:
            logger.error(f"Error shutting down middleware pipeline: {e}")
    
    async def health_check(self) -> bool:
        """Verificar salud del pipeline."""
        return self._initialized
    
    @property
    def name(self) -> str:
        return f"MiddlewarePipeline_{self.name}"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    async def add_middleware(self, middleware: IMiddleware, middleware_type: MiddlewareType, 
                           priority: int = 0, enabled: bool = True) -> bool:
        """Agregar middleware al pipeline."""
        try:
            async with self._lock:
                node = MiddlewareNode(middleware, middleware_type, priority, enabled)
                self.nodes.append(node)
                
                # Ordenar por prioridad (mayor prioridad primero)
                self.nodes.sort(key=lambda x: x.priority, reverse=True)
                
                logger.info(f"Added middleware {middleware.__class__.__name__} to pipeline {self.name}")
                return True
                
        except Exception as e:
            logger.error(f"Error adding middleware to pipeline: {e}")
            return False
    
    async def remove_middleware(self, middleware: IMiddleware) -> bool:
        """Remover middleware del pipeline."""
        try:
            async with self._lock:
                for i, node in enumerate(self.nodes):
                    if node.middleware == middleware:
                        del self.nodes[i]
                        logger.info(f"Removed middleware {middleware.__class__.__name__} from pipeline {self.name}")
                        return True
                return False
                
        except Exception as e:
            logger.error(f"Error removing middleware from pipeline: {e}")
            return False
    
    async def enable_middleware(self, middleware: IMiddleware) -> bool:
        """Habilitar middleware."""
        try:
            async with self._lock:
                for node in self.nodes:
                    if node.middleware == middleware:
                        node.enabled = True
                        logger.info(f"Enabled middleware {middleware.__class__.__name__}")
                        return True
                return False
                
        except Exception as e:
            logger.error(f"Error enabling middleware: {e}")
            return False
    
    async def disable_middleware(self, middleware: IMiddleware) -> bool:
        """Deshabilitar middleware."""
        try:
            async with self._lock:
                for node in self.nodes:
                    if node.middleware == middleware:
                        node.enabled = False
                        logger.info(f"Disabled middleware {middleware.__class__.__name__}")
                        return True
                return False
                
        except Exception as e:
            logger.error(f"Error disabling middleware: {e}")
            return False
    
    async def execute_pipeline(self, context: MiddlewareContext) -> MiddlewareContext:
        """Ejecutar pipeline completo."""
        try:
            start_time = datetime.utcnow()
            
            # Ejecutar middleware de request primero
            request_nodes = [node for node in self.nodes 
                           if node.middleware_type == MiddlewareType.REQUEST and node.enabled]
            
            for node in request_nodes:
                context = await node.execute(context)
                if context.has_errors():
                    logger.warning(f"Request middleware {node.middleware.__class__.__name__} produced errors")
            
            # Ejecutar otros tipos de middleware
            other_nodes = [node for node in self.nodes 
                          if node.middleware_type != MiddlewareType.REQUEST and 
                          node.middleware_type != MiddlewareType.RESPONSE and node.enabled]
            
            for node in other_nodes:
                context = await node.execute(context)
                if context.has_errors():
                    logger.warning(f"Middleware {node.middleware.__class__.__name__} produced errors")
            
            # Ejecutar middleware de response al final
            response_nodes = [node for node in self.nodes 
                            if node.middleware_type == MiddlewareType.RESPONSE and node.enabled]
            
            for node in response_nodes:
                context = await node.execute(context)
                if context.has_errors():
                    logger.warning(f"Response middleware {node.middleware.__class__.__name__} produced errors")
            
            # Actualizar estadísticas
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            await self._update_execution_stats(execution_time, context.has_errors())
            
            return context
            
        except Exception as e:
            logger.error(f"Error executing middleware pipeline: {e}")
            context.add_error(e)
            return context
    
    async def execute_request_pipeline(self, request: Any) -> MiddlewareContext:
        """Ejecutar pipeline de request."""
        try:
            context = MiddlewareContext(request=request)
            return await self.execute_pipeline(context)
            
        except Exception as e:
            logger.error(f"Error executing request pipeline: {e}")
            context = MiddlewareContext(request=request)
            context.add_error(e)
            return context
    
    async def execute_response_pipeline(self, response: Any) -> MiddlewareContext:
        """Ejecutar pipeline de response."""
        try:
            context = MiddlewareContext(response=response)
            return await self.execute_pipeline(context)
            
        except Exception as e:
            logger.error(f"Error executing response pipeline: {e}")
            context = MiddlewareContext(response=response)
            context.add_error(e)
            return context
    
    async def _update_execution_stats(self, execution_time: float, has_errors: bool) -> None:
        """Actualizar estadísticas de ejecución."""
        try:
            if "total_executions" not in self._execution_stats:
                self._execution_stats = {
                    "total_executions": 0,
                    "successful_executions": 0,
                    "failed_executions": 0,
                    "total_execution_time": 0.0,
                    "average_execution_time": 0.0,
                    "last_execution": None
                }
            
            stats = self._execution_stats
            stats["total_executions"] += 1
            stats["total_execution_time"] += execution_time
            stats["average_execution_time"] = stats["total_execution_time"] / stats["total_executions"]
            stats["last_execution"] = datetime.utcnow().isoformat()
            
            if has_errors:
                stats["failed_executions"] += 1
            else:
                stats["successful_executions"] += 1
                
        except Exception as e:
            logger.error(f"Error updating execution stats: {e}")
    
    async def get_pipeline_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del pipeline."""
        try:
            node_stats = []
            for node in self.nodes:
                node_stats.append(node.get_stats())
            
            return {
                "pipeline_name": self.name,
                "node_count": len(self.nodes),
                "enabled_nodes": len([node for node in self.nodes if node.enabled]),
                "disabled_nodes": len([node for node in self.nodes if not node.enabled]),
                "execution_stats": self._execution_stats,
                "node_stats": node_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting pipeline stats: {e}")
            return {}
    
    async def get_middleware_by_type(self, middleware_type: MiddlewareType) -> List[MiddlewareNode]:
        """Obtener middleware por tipo."""
        return [node for node in self.nodes if node.middleware_type == middleware_type]
    
    async def get_middleware_by_priority(self, min_priority: int = None, max_priority: int = None) -> List[MiddlewareNode]:
        """Obtener middleware por rango de prioridad."""
        filtered_nodes = self.nodes
        
        if min_priority is not None:
            filtered_nodes = [node for node in filtered_nodes if node.priority >= min_priority]
        
        if max_priority is not None:
            filtered_nodes = [node for node in filtered_nodes if node.priority <= max_priority]
        
        return filtered_nodes
    
    async def reorder_middleware(self, middleware: IMiddleware, new_priority: int) -> bool:
        """Reordenar middleware por prioridad."""
        try:
            async with self._lock:
                for node in self.nodes:
                    if node.middleware == middleware:
                        node.priority = new_priority
                        # Reordenar por prioridad
                        self.nodes.sort(key=lambda x: x.priority, reverse=True)
                        logger.info(f"Reordered middleware {middleware.__class__.__name__} to priority {new_priority}")
                        return True
                return False
                
        except Exception as e:
            logger.error(f"Error reordering middleware: {e}")
            return False
    
    async def clear_pipeline(self) -> None:
        """Limpiar pipeline."""
        try:
            async with self._lock:
                self.nodes.clear()
                self._execution_stats.clear()
                logger.info(f"Cleared middleware pipeline {self.name}")
                
        except Exception as e:
            logger.error(f"Error clearing pipeline: {e}")
    
    async def validate_pipeline(self) -> Dict[str, Any]:
        """Validar pipeline."""
        try:
            validation_results = []
            
            for node in self.nodes:
                try:
                    # Verificar que el middleware implementa los métodos requeridos
                    if not hasattr(node.middleware, 'process_request'):
                        validation_results.append({
                            "middleware": node.middleware.__class__.__name__,
                            "valid": False,
                            "error": "Missing process_request method"
                        })
                        continue
                    
                    # Verificar que el middleware está habilitado
                    if not node.enabled:
                        validation_results.append({
                            "middleware": node.middleware.__class__.__name__,
                            "valid": False,
                            "error": "Middleware is disabled"
                        })
                        continue
                    
                    validation_results.append({
                        "middleware": node.middleware.__class__.__name__,
                        "valid": True
                    })
                    
                except Exception as e:
                    validation_results.append({
                        "middleware": node.middleware.__class__.__name__,
                        "valid": False,
                        "error": str(e)
                    })
            
            return {
                "valid": all(result["valid"] for result in validation_results),
                "pipeline_name": self.name,
                "node_count": len(self.nodes),
                "validation_results": validation_results
            }
            
        except Exception as e:
            logger.error(f"Error validating pipeline: {e}")
            return {"valid": False, "error": str(e)}
    
    async def clone_pipeline(self, new_name: str) -> 'MiddlewarePipeline':
        """Clonar pipeline."""
        try:
            new_pipeline = MiddlewarePipeline(new_name)
            await new_pipeline.initialize()
            
            # Copiar nodos
            for node in self.nodes:
                await new_pipeline.add_middleware(
                    node.middleware, 
                    node.middleware_type, 
                    node.priority, 
                    node.enabled
                )
            
            logger.info(f"Cloned pipeline {self.name} to {new_name}")
            return new_pipeline
            
        except Exception as e:
            logger.error(f"Error cloning pipeline: {e}")
            raise




