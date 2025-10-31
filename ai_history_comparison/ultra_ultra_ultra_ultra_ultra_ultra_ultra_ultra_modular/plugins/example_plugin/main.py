"""
Example Plugin - Plugin de Ejemplo
=================================

Plugin de ejemplo que demuestra cómo crear plugins para el sistema ultra modular.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from ...core.interfaces.base_interfaces import IPlugin, IAnalyzer, ITransformer, IExtension

logger = logging.getLogger(__name__)


class ExamplePlugin(IPlugin):
    """Plugin de ejemplo."""
    
    def __init__(self):
        self._installed = False
        self._active = False
        self._config = {}
    
    async def install(self) -> bool:
        """Instalar plugin."""
        try:
            logger.info("Installing Example Plugin...")
            
            # Configuración por defecto
            self._config = {
                "debug_mode": False,
                "max_retries": 3,
                "timeout_seconds": 30
            }
            
            self._installed = True
            logger.info("Example Plugin installed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error installing Example Plugin: {e}")
            return False
    
    async def uninstall(self) -> bool:
        """Desinstalar plugin."""
        try:
            logger.info("Uninstalling Example Plugin...")
            
            if self._active:
                await self.deactivate()
            
            self._installed = False
            logger.info("Example Plugin uninstalled successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error uninstalling Example Plugin: {e}")
            return False
    
    async def activate(self) -> bool:
        """Activar plugin."""
        try:
            if not self._installed:
                logger.error("Plugin not installed")
                return False
            
            logger.info("Activating Example Plugin...")
            
            # Lógica de activación
            self._active = True
            logger.info("Example Plugin activated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error activating Example Plugin: {e}")
            return False
    
    async def deactivate(self) -> bool:
        """Desactivar plugin."""
        try:
            logger.info("Deactivating Example Plugin...")
            
            # Lógica de desactivación
            self._active = False
            logger.info("Example Plugin deactivated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deactivating Example Plugin: {e}")
            return False
    
    @property
    def plugin_info(self) -> Dict[str, Any]:
        """Información del plugin."""
        return {
            "name": "example_plugin",
            "version": "1.0.0",
            "description": "Plugin de ejemplo para demostrar el sistema de plugins",
            "author": "AI History Comparison Team",
            "installed": self._installed,
            "active": self._active,
            "config": self._config
        }


class ExampleAnalyzer(IAnalyzer):
    """Analizador de ejemplo."""
    
    def __init__(self):
        self._analysis_type = "example_analysis"
    
    async def analyze(self, content: str) -> Dict[str, Any]:
        """Analizar contenido."""
        try:
            logger.info(f"Analyzing content with Example Analyzer: {len(content)} characters")
            
            # Análisis simple de ejemplo
            analysis = {
                "analyzer": "example_analyzer",
                "content_length": len(content),
                "word_count": len(content.split()),
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "example_metric": len(content) * 0.1,  # Métrica de ejemplo
                "analysis_type": self._analysis_type
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in Example Analyzer: {e}")
            return {"error": str(e)}
    
    async def get_analysis_type(self) -> str:
        """Obtener tipo de análisis."""
        return self._analysis_type
    
    async def process(self, data: Any) -> Any:
        """Procesar datos."""
        if isinstance(data, str):
            return await self.analyze(data)
        return data
    
    async def validate(self, data: Any) -> bool:
        """Validar datos."""
        return isinstance(data, str) and len(data) > 0


class ExampleTransformer(ITransformer):
    """Transformador de ejemplo."""
    
    def __init__(self):
        self._transformation_type = "example_transformation"
    
    async def transform(self, data: Any) -> Any:
        """Transformar datos."""
        try:
            logger.info(f"Transforming data with Example Transformer")
            
            if isinstance(data, str):
                # Transformación simple: agregar prefijo
                return f"[EXAMPLE] {data}"
            elif isinstance(data, dict):
                # Transformación de diccionario
                transformed = data.copy()
                transformed["transformed_by"] = "example_transformer"
                transformed["transformation_timestamp"] = datetime.utcnow().isoformat()
                return transformed
            
            return data
            
        except Exception as e:
            logger.error(f"Error in Example Transformer: {e}")
            return data
    
    async def get_transformation_type(self) -> str:
        """Obtener tipo de transformación."""
        return self._transformation_type


class ExampleExtension(IExtension):
    """Extensión de ejemplo."""
    
    def __init__(self, priority: int = 0):
        self._priority = priority
        self._installed = False
        self._active = False
    
    async def install(self) -> bool:
        """Instalar extensión."""
        try:
            logger.info("Installing Example Extension...")
            self._installed = True
            return True
        except Exception as e:
            logger.error(f"Error installing Example Extension: {e}")
            return False
    
    async def uninstall(self) -> bool:
        """Desinstalar extensión."""
        try:
            logger.info("Uninstalling Example Extension...")
            if self._active:
                await self.deactivate()
            self._installed = False
            return True
        except Exception as e:
            logger.error(f"Error uninstalling Example Extension: {e}")
            return False
    
    async def activate(self) -> bool:
        """Activar extensión."""
        try:
            logger.info("Activating Example Extension...")
            self._active = True
            return True
        except Exception as e:
            logger.error(f"Error activating Example Extension: {e}")
            return False
    
    async def deactivate(self) -> bool:
        """Desactivar extensión."""
        try:
            logger.info("Deactivating Example Extension...")
            self._active = False
            return True
        except Exception as e:
            logger.error(f"Error deactivating Example Extension: {e}")
            return False
    
    @property
    def plugin_info(self) -> Dict[str, Any]:
        """Información del plugin."""
        return {
            "name": "example_extension",
            "version": "1.0.0",
            "description": "Extensión de ejemplo",
            "priority": self._priority,
            "installed": self._installed,
            "active": self._active
        }
    
    async def extend(self, target: Any) -> Any:
        """Extender funcionalidad."""
        try:
            logger.info("Extending functionality with Example Extension...")
            
            if isinstance(target, str):
                return f"[EXTENDED] {target}"
            elif isinstance(target, dict):
                extended = target.copy()
                extended["extended_by"] = "example_extension"
                extended["extension_timestamp"] = datetime.utcnow().isoformat()
                return extended
            
            return target
            
        except Exception as e:
            logger.error(f"Error in Example Extension: {e}")
            return target
    
    async def get_extension_points(self) -> List[str]:
        """Obtener puntos de extensión."""
        return ["pre_analysis", "post_analysis", "content_transformation"]


# Instancias globales para el plugin
example_plugin = ExamplePlugin()
example_analyzer = ExampleAnalyzer()
example_transformer = ExampleTransformer()
example_extension = ExampleExtension(priority=100)




