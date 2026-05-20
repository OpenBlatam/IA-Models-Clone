"""
Plugin System - Sistema de plugins
===================================

Sistema extensible de plugins para agregar funcionalidades al chat.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PluginType(Enum):
    """Tipos de plugins."""
    PRE_PROCESSOR = "pre_processor"  # Procesar mensaje antes de enviar al LLM
    POST_PROCESSOR = "post_processor"  # Procesar respuesta después del LLM
    MIDDLEWARE = "middleware"  # Interceptar y modificar el flujo
    ANALYZER = "analyzer"  # Analizar conversaciones
    NOTIFICATION = "notification"  # Notificaciones y alertas


@dataclass
class PluginContext:
    """Contexto para plugins."""
    session_id: str
    user_id: Optional[str]
    message: Optional[str] = None
    response: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BasePlugin(ABC):
    """Clase base para plugins."""
    
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.config: Dict[str, Any] = {}
    
    @abstractmethod
    async def execute(self, context: PluginContext) -> PluginContext:
        """
        Ejecutar el plugin.
        
        Args:
            context: Contexto del plugin
        
        Returns:
            Contexto modificado
        """
        pass
    
    def configure(self, config: Dict[str, Any]):
        """Configurar el plugin."""
        self.config.update(config)
        logger.info(f"Plugin {self.name} configured: {config}")
    
    async def initialize(self):
        """Inicializar el plugin (opcional)."""
        pass
    
    async def cleanup(self):
        """Limpiar recursos del plugin (opcional)."""
        pass


class PluginManager:
    """Gestor de plugins."""
    
    def __init__(self):
        self.plugins: Dict[PluginType, List[BasePlugin]] = {
            plugin_type: []
            for plugin_type in PluginType
        }
        self._lock = asyncio.Lock()
    
    def register(self, plugin: BasePlugin, plugin_type: PluginType):
        """Registrar un plugin."""
        async with self._lock:
            if plugin.enabled:
                self.plugins[plugin_type].append(plugin)
                logger.info(f"Registered plugin: {plugin.name} ({plugin_type.value})")
                
                # Inicializar si es necesario
                try:
                    await plugin.initialize()
                except Exception as e:
                    logger.error(f"Error initializing plugin {plugin.name}: {e}")
    
    def unregister(self, plugin_name: str, plugin_type: PluginType):
        """Desregistrar un plugin."""
        async with self._lock:
            self.plugins[plugin_type] = [
                p for p in self.plugins[plugin_type]
                if p.name != plugin_name
            ]
            logger.info(f"Unregistered plugin: {plugin_name}")
    
    async def execute_plugins(
        self,
        plugin_type: PluginType,
        context: PluginContext,
    ) -> PluginContext:
        """Ejecutar todos los plugins de un tipo."""
        async with self._lock:
            plugins = self.plugins[plugin_type]
        
        for plugin in plugins:
            if not plugin.enabled:
                continue
            
            try:
                context = await plugin.execute(context)
            except Exception as e:
                logger.error(f"Error executing plugin {plugin.name}: {e}")
                # Continuar con otros plugins aunque uno falle
        
        return context
    
    def get_plugins(self, plugin_type: Optional[PluginType] = None) -> List[BasePlugin]:
        """Obtener plugins registrados."""
        if plugin_type:
            return self.plugins[plugin_type]
        else:
            return [
                plugin
                for plugins_list in self.plugins.values()
                for plugin in plugins_list
            ]
    
    async def cleanup_all(self):
        """Limpiar todos los plugins."""
        async with self._lock:
            for plugins in self.plugins.values():
                for plugin in plugins:
                    try:
                        await plugin.cleanup()
                    except Exception as e:
                        logger.error(f"Error cleaning up plugin {plugin.name}: {e}")


# Plugins de ejemplo

class SentimentAnalyzerPlugin(BasePlugin):
    """Plugin para analizar sentimiento de mensajes."""
    
    def __init__(self):
        super().__init__("sentiment_analyzer")
    
    async def execute(self, context: PluginContext) -> PluginContext:
        """Analizar sentimiento del mensaje."""
        if context.message:
            # Análisis simple (en producción usaría un modelo de NLP)
            positive_words = ["bueno", "excelente", "perfecto", "genial", "me gusta"]
            negative_words = ["malo", "terrible", "horrible", "no me gusta"]
            
            message_lower = context.message.lower()
            positive_count = sum(1 for word in positive_words if word in message_lower)
            negative_count = sum(1 for word in negative_words if word in message_lower)
            
            if positive_count > negative_count:
                sentiment = "positive"
            elif negative_count > positive_count:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            context.metadata["sentiment"] = sentiment
            context.metadata["sentiment_score"] = positive_count - negative_count
        
        return context


class ProfanityFilterPlugin(BasePlugin):
    """Plugin para filtrar contenido ofensivo."""
    
    def __init__(self):
        super().__init__("profanity_filter")
        self.blocked_words = ["palabra1", "palabra2"]  # En producción desde config
    
    async def execute(self, context: PluginContext) -> PluginContext:
        """Filtrar contenido ofensivo."""
        if context.message:
            message_lower = context.message.lower()
            for word in self.blocked_words:
                if word in message_lower:
                    context.message = context.message.replace(word, "*" * len(word))
                    context.metadata["filtered"] = True
        
        return context


class ResponseEnhancerPlugin(BasePlugin):
    """Plugin para mejorar respuestas."""
    
    def __init__(self):
        super().__init__("response_enhancer")
    
    async def execute(self, context: PluginContext) -> PluginContext:
        """Mejorar respuesta."""
        if context.response:
            # Agregar formato, emojis, etc.
            # Por ahora, solo agregar metadata
            context.metadata["enhanced"] = True
        
        return context


class NotificationPlugin(BasePlugin):
    """Plugin para enviar notificaciones."""
    
    def __init__(self):
        super().__init__("notification")
    
    async def execute(self, context: PluginContext) -> PluginContext:
        """Enviar notificación."""
        # En producción, enviar notificación real
        logger.info(f"Notification: {context.session_id} - {context.message}")
        return context
































