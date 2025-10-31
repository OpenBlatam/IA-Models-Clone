"""
Extension Manager - Gestor de Extensiones
========================================

Sistema de gestión de extensiones dinámicas con puntos de extensión configurables.
"""

import asyncio
import inspect
import logging
from typing import Dict, List, Any, Optional, Type, Callable, Set
from datetime import datetime
from enum import Enum

from ..interfaces.base_interfaces import IExtension, IComponent, IRegistry

logger = logging.getLogger(__name__)


class ExtensionPointType(Enum):
    """Tipos de puntos de extensión."""
    PRE_PROCESS = "pre_process"
    POST_PROCESS = "post_process"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    FILTER = "filter"
    AGGREGATION = "aggregation"
    NOTIFICATION = "notification"
    CUSTOM = "custom"


class ExtensionPoint:
    """Punto de extensión."""
    
    def __init__(self, name: str, point_type: ExtensionPointType, 
                 description: str = "", priority: int = 0):
        self.name = name
        self.point_type = point_type
        self.description = description
        self.priority = priority
        self.extensions: List[IExtension] = []
        self.created_at = datetime.utcnow()
    
    def add_extension(self, extension: IExtension) -> None:
        """Agregar extensión al punto."""
        self.extensions.append(extension)
        # Ordenar por prioridad
        self.extensions.sort(key=lambda x: getattr(x, 'priority', 0), reverse=True)
    
    def remove_extension(self, extension: IExtension) -> bool:
        """Remover extensión del punto."""
        if extension in self.extensions:
            self.extensions.remove(extension)
            return True
        return False
    
    def get_extensions(self) -> List[IExtension]:
        """Obtener extensiones ordenadas por prioridad."""
        return sorted(self.extensions, key=lambda x: getattr(x, 'priority', 0), reverse=True)


class ExtensionContext:
    """Contexto de extensión."""
    
    def __init__(self, data: Any, metadata: Dict[str, Any] = None):
        self.data = data
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
        self.modified = False
    
    def set_data(self, data: Any) -> None:
        """Establecer datos."""
        self.data = data
        self.modified = True
    
    def get_data(self) -> Any:
        """Obtener datos."""
        return self.data
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Establecer metadatos."""
        self.metadata[key] = value
        self.modified = True
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Obtener metadatos."""
        return self.metadata.get(key, default)


class ExtensionRegistry(IRegistry):
    """Registro de extensiones."""
    
    def __init__(self):
        self._extensions: Dict[str, IExtension] = {}
        self._extension_points: Dict[str, ExtensionPoint] = {}
        self._lock = asyncio.Lock()
    
    async def register(self, name: str, component: Any) -> bool:
        """Registrar extensión."""
        try:
            async with self._lock:
                if isinstance(component, IExtension):
                    self._extensions[name] = component
                    return True
                return False
        except Exception as e:
            logger.error(f"Error registering extension {name}: {e}")
            return False
    
    async def unregister(self, name: str) -> bool:
        """Desregistrar extensión."""
        try:
            async with self._lock:
                if name in self._extensions:
                    extension = self._extensions[name]
                    
                    # Remover de todos los puntos de extensión
                    for point in self._extension_points.values():
                        point.remove_extension(extension)
                    
                    del self._extensions[name]
                    return True
                return False
        except Exception as e:
            logger.error(f"Error unregistering extension {name}: {e}")
            return False
    
    async def get(self, name: str) -> Optional[Any]:
        """Obtener extensión."""
        return self._extensions.get(name)
    
    async def list_all(self) -> List[str]:
        """Listar todas las extensiones."""
        return list(self._extensions.keys())
    
    async def register_extension_point(self, point: ExtensionPoint) -> bool:
        """Registrar punto de extensión."""
        try:
            async with self._lock:
                self._extension_points[point.name] = point
                return True
        except Exception as e:
            logger.error(f"Error registering extension point {point.name}: {e}")
            return False
    
    async def get_extension_point(self, name: str) -> Optional[ExtensionPoint]:
        """Obtener punto de extensión."""
        return self._extension_points.get(name)
    
    async def list_extension_points(self) -> List[str]:
        """Listar puntos de extensión."""
        return list(self._extension_points.keys())


class ExtensionManager(IComponent):
    """Gestor principal de extensiones."""
    
    def __init__(self):
        self.registry = ExtensionRegistry()
        self._initialized = False
        self._execution_stats: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self) -> None:
        """Inicializar gestor de extensiones."""
        try:
            # Crear puntos de extensión por defecto
            await self._create_default_extension_points()
            
            self._initialized = True
            logger.info("Extension manager initialized")
            
        except Exception as e:
            logger.error(f"Error initializing extension manager: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Cerrar gestor de extensiones."""
        try:
            # Desactivar todas las extensiones
            extensions = await self.registry.list_all()
            for extension_name in extensions:
                extension = await self.registry.get(extension_name)
                if extension:
                    await extension.deactivate()
            
            self._initialized = False
            logger.info("Extension manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error shutting down extension manager: {e}")
    
    async def health_check(self) -> bool:
        """Verificar salud del gestor."""
        return self._initialized
    
    @property
    def name(self) -> str:
        return "ExtensionManager"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    async def _create_default_extension_points(self) -> None:
        """Crear puntos de extensión por defecto."""
        default_points = [
            ExtensionPoint("pre_analysis", ExtensionPointType.PRE_PROCESS, 
                          "Antes del análisis de contenido", priority=100),
            ExtensionPoint("post_analysis", ExtensionPointType.POST_PROCESS, 
                          "Después del análisis de contenido", priority=100),
            ExtensionPoint("pre_comparison", ExtensionPointType.PRE_PROCESS, 
                          "Antes de la comparación", priority=100),
            ExtensionPoint("post_comparison", ExtensionPointType.POST_PROCESS, 
                          "Después de la comparación", priority=100),
            ExtensionPoint("pre_quality_assessment", ExtensionPointType.PRE_PROCESS, 
                          "Antes de la evaluación de calidad", priority=100),
            ExtensionPoint("post_quality_assessment", ExtensionPointType.POST_PROCESS, 
                          "Después de la evaluación de calidad", priority=100),
            ExtensionPoint("content_validation", ExtensionPointType.VALIDATION, 
                          "Validación de contenido", priority=50),
            ExtensionPoint("content_transformation", ExtensionPointType.TRANSFORMATION, 
                          "Transformación de contenido", priority=50),
            ExtensionPoint("result_filtering", ExtensionPointType.FILTER, 
                          "Filtrado de resultados", priority=30),
            ExtensionPoint("metrics_aggregation", ExtensionPointType.AGGREGATION, 
                          "Agregación de métricas", priority=20),
            ExtensionPoint("notification", ExtensionPointType.NOTIFICATION, 
                          "Notificaciones", priority=10)
        ]
        
        for point in default_points:
            await self.registry.register_extension_point(point)
    
    async def register_extension(self, name: str, extension: IExtension) -> bool:
        """Registrar extensión."""
        try:
            if await self.registry.register(name, extension):
                # Instalar y activar extensión
                if await extension.install():
                    if await extension.activate():
                        logger.info(f"Extension {name} registered and activated")
                        return True
                    else:
                        logger.error(f"Failed to activate extension {name}")
                        return False
                else:
                    logger.error(f"Failed to install extension {name}")
                    return False
            return False
            
        except Exception as e:
            logger.error(f"Error registering extension {name}: {e}")
            return False
    
    async def unregister_extension(self, name: str) -> bool:
        """Desregistrar extensión."""
        try:
            extension = await self.registry.get(name)
            if extension:
                # Desactivar y desinstalar
                await extension.deactivate()
                await extension.uninstall()
                
                # Remover de puntos de extensión
                for point_name in await self.registry.list_extension_points():
                    point = await self.registry.get_extension_point(point_name)
                    if point:
                        point.remove_extension(extension)
                
                return await self.registry.unregister(name)
            
            return False
            
        except Exception as e:
            logger.error(f"Error unregistering extension {name}: {e}")
            return False
    
    async def create_extension_point(self, name: str, point_type: ExtensionPointType, 
                                   description: str = "", priority: int = 0) -> bool:
        """Crear punto de extensión."""
        try:
            point = ExtensionPoint(name, point_type, description, priority)
            return await self.registry.register_extension_point(point)
            
        except Exception as e:
            logger.error(f"Error creating extension point {name}: {e}")
            return False
    
    async def attach_extension_to_point(self, extension_name: str, point_name: str) -> bool:
        """Adjuntar extensión a punto de extensión."""
        try:
            extension = await self.registry.get(extension_name)
            point = await self.registry.get_extension_point(point_name)
            
            if not extension:
                logger.error(f"Extension {extension_name} not found")
                return False
            
            if not point:
                logger.error(f"Extension point {point_name} not found")
                return False
            
            point.add_extension(extension)
            logger.info(f"Extension {extension_name} attached to point {point_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error attaching extension {extension_name} to point {point_name}: {e}")
            return False
    
    async def detach_extension_from_point(self, extension_name: str, point_name: str) -> bool:
        """Desadjuntar extensión de punto de extensión."""
        try:
            extension = await self.registry.get(extension_name)
            point = await self.registry.get_extension_point(point_name)
            
            if not extension or not point:
                return False
            
            return point.remove_extension(extension)
            
        except Exception as e:
            logger.error(f"Error detaching extension {extension_name} from point {point_name}: {e}")
            return False
    
    async def execute_extensions(self, point_name: str, context: ExtensionContext) -> ExtensionContext:
        """Ejecutar extensiones en un punto de extensión."""
        try:
            point = await self.registry.get_extension_point(point_name)
            if not point:
                logger.warning(f"Extension point {point_name} not found")
                return context
            
            extensions = point.get_extensions()
            if not extensions:
                return context
            
            # Ejecutar extensiones en orden de prioridad
            for extension in extensions:
                try:
                    # Verificar si la extensión puede manejar el contexto
                    if hasattr(extension, 'can_extend') and not await extension.can_extend(context.data):
                        continue
                    
                    # Ejecutar extensión
                    result = await extension.extend(context.data)
                    
                    # Actualizar contexto si la extensión devuelve resultado
                    if result is not None:
                        context.set_data(result)
                    
                    # Actualizar estadísticas
                    await self._update_execution_stats(point_name, extension, True)
                    
                except Exception as e:
                    logger.error(f"Error executing extension {extension.__class__.__name__} at point {point_name}: {e}")
                    await self._update_execution_stats(point_name, extension, False)
            
            return context
            
        except Exception as e:
            logger.error(f"Error executing extensions at point {point_name}: {e}")
            return context
    
    async def _update_execution_stats(self, point_name: str, extension: IExtension, success: bool) -> None:
        """Actualizar estadísticas de ejecución."""
        try:
            extension_name = extension.__class__.__name__
            stats_key = f"{point_name}:{extension_name}"
            
            if stats_key not in self._execution_stats:
                self._execution_stats[stats_key] = {
                    "total_executions": 0,
                    "successful_executions": 0,
                    "failed_executions": 0,
                    "last_execution": None
                }
            
            stats = self._execution_stats[stats_key]
            stats["total_executions"] += 1
            stats["last_execution"] = datetime.utcnow()
            
            if success:
                stats["successful_executions"] += 1
            else:
                stats["failed_executions"] += 1
                
        except Exception as e:
            logger.error(f"Error updating execution stats: {e}")
    
    async def get_extension_point_info(self, point_name: str) -> Optional[Dict[str, Any]]:
        """Obtener información de punto de extensión."""
        try:
            point = await self.registry.get_extension_point(point_name)
            if not point:
                return None
            
            extensions = point.get_extensions()
            extension_info = []
            
            for extension in extensions:
                extension_info.append({
                    "name": extension.__class__.__name__,
                    "priority": getattr(extension, 'priority', 0),
                    "plugin_info": extension.plugin_info if hasattr(extension, 'plugin_info') else {}
                })
            
            return {
                "name": point.name,
                "type": point.point_type.value,
                "description": point.description,
                "priority": point.priority,
                "created_at": point.created_at.isoformat(),
                "extensions": extension_info,
                "extension_count": len(extensions)
            }
            
        except Exception as e:
            logger.error(f"Error getting extension point info for {point_name}: {e}")
            return None
    
    async def list_extension_points(self) -> List[Dict[str, Any]]:
        """Listar todos los puntos de extensión."""
        try:
            points = []
            for point_name in await self.registry.list_extension_points():
                point_info = await self.get_extension_point_info(point_name)
                if point_info:
                    points.append(point_info)
            return points
            
        except Exception as e:
            logger.error(f"Error listing extension points: {e}")
            return []
    
    async def get_execution_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de ejecución."""
        return self._execution_stats.copy()
    
    async def get_extension_stats(self, extension_name: str) -> Dict[str, Any]:
        """Obtener estadísticas de una extensión específica."""
        try:
            extension_stats = {}
            for stats_key, stats in self._execution_stats.items():
                if extension_name in stats_key:
                    extension_stats[stats_key] = stats
            
            return extension_stats
            
        except Exception as e:
            logger.error(f"Error getting extension stats for {extension_name}: {e}")
            return {}
    
    async def clear_execution_stats(self) -> None:
        """Limpiar estadísticas de ejecución."""
        self._execution_stats.clear()
    
    async def get_extensions_for_point(self, point_name: str) -> List[Dict[str, Any]]:
        """Obtener extensiones para un punto específico."""
        try:
            point = await self.registry.get_extension_point(point_name)
            if not point:
                return []
            
            extensions = []
            for extension in point.get_extensions():
                extensions.append({
                    "name": extension.__class__.__name__,
                    "priority": getattr(extension, 'priority', 0),
                    "plugin_info": extension.plugin_info if hasattr(extension, 'plugin_info') else {}
                })
            
            return extensions
            
        except Exception as e:
            logger.error(f"Error getting extensions for point {point_name}: {e}")
            return []
    
    async def validate_extension_point(self, point_name: str) -> Dict[str, Any]:
        """Validar punto de extensión."""
        try:
            point = await self.registry.get_extension_point(point_name)
            if not point:
                return {"valid": False, "error": "Extension point not found"}
            
            extensions = point.get_extensions()
            validation_results = []
            
            for extension in extensions:
                try:
                    # Verificar que la extensión implementa los métodos requeridos
                    if not hasattr(extension, 'extend'):
                        validation_results.append({
                            "extension": extension.__class__.__name__,
                            "valid": False,
                            "error": "Missing extend method"
                        })
                        continue
                    
                    # Verificar que la extensión está activa
                    if hasattr(extension, 'is_active') and not await extension.is_active():
                        validation_results.append({
                            "extension": extension.__class__.__name__,
                            "valid": False,
                            "error": "Extension not active"
                        })
                        continue
                    
                    validation_results.append({
                        "extension": extension.__class__.__name__,
                        "valid": True
                    })
                    
                except Exception as e:
                    validation_results.append({
                        "extension": extension.__class__.__name__,
                        "valid": False,
                        "error": str(e)
                    })
            
            return {
                "valid": all(result["valid"] for result in validation_results),
                "point_name": point_name,
                "extension_count": len(extensions),
                "validation_results": validation_results
            }
            
        except Exception as e:
            logger.error(f"Error validating extension point {point_name}: {e}")
            return {"valid": False, "error": str(e)}




