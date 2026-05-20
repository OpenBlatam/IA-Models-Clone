"""
Event Service - Servicio de eventos para arquitectura event-driven
==================================================================

Servicio de eventos que abstrae el message broker utilizado.
"""

import logging
from typing import Optional, Dict, Any, Callable
from abc import ABC, abstractmethod

from ..core.message_broker import get_message_broker
from ..core.microservices_config import get_microservices_config

logger = logging.getLogger(__name__)


class EventPublisher:
    """
    Publicador de eventos.
    
    Abstrae el message broker utilizado para publicar eventos
    en arquitectura event-driven.
    """
    
    def __init__(self):
        config = get_microservices_config()
        if config.message_broker_type.value != "none":
            try:
                self.broker = get_message_broker()
            except Exception as e:
                logger.warning(f"Message broker not available: {e}")
                self.broker = None
        else:
            self.broker = None
    
    async def publish(
        self,
        event_type: str,
        event_data: Dict[str, Any]
    ) -> bool:
        """
        Publica un evento.
        
        Args:
            event_type: Tipo de evento (ej: "project.created")
            event_data: Datos del evento
        
        Returns:
            True si se publicó exitosamente
        """
        if not self.broker:
            logger.debug(f"Event broker not available, skipping: {event_type}")
            return False
        
        try:
            event = {
                "type": event_type,
                "data": event_data,
                "timestamp": str(logging.time.time())
            }
            
            # Extraer tópico del tipo de evento (ej: "project.created" -> "project")
            topic = event_type.split(".")[0]
            
            return await self.broker.publish(topic, event)
        except Exception as e:
            logger.error(f"Error publishing event {event_type}: {e}")
            return False
    
    async def publish_batch(
        self,
        events: list
    ) -> Dict[str, bool]:
        """
        Publica múltiples eventos.
        
        Args:
            events: Lista de tuplas (event_type, event_data)
        
        Returns:
            Diccionario con resultados por evento
        """
        results = {}
        for event_type, event_data in events:
            results[event_type] = await self.publish(event_type, event_data)
        return results


class EventSubscriber:
    """
    Suscriptor de eventos.
    
    Permite suscribirse a eventos y ejecutar callbacks.
    """
    
    def __init__(self):
        config = get_microservices_config()
        if config.message_broker_type.value != "none":
            try:
                self.broker = get_message_broker()
            except Exception as e:
                logger.warning(f"Message broker not available: {e}")
                self.broker = None
        else:
            self.broker = None
        
        self._handlers: Dict[str, list] = {}
    
    def subscribe(
        self,
        event_type: str,
        handler: Callable[[Dict[str, Any]], None]
    ):
        """
        Suscribe un handler a un tipo de evento.
        
        Args:
            event_type: Tipo de evento (ej: "project.created")
            handler: Función que maneja el evento
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        
        self._handlers[event_type].append(handler)
    
    async def start_listening(self):
        """Inicia escucha de eventos"""
        if not self.broker:
            logger.warning("Event broker not available, cannot start listening")
            return
        
        # Suscribirse a todos los tópicos relevantes
        topics = set(event_type.split(".")[0] for event_type in self._handlers.keys())
        
        for topic in topics:
            await self.broker.subscribe(topic, self._handle_event)
    
    async def _handle_event(self, event: Dict[str, Any]):
        """Maneja un evento recibido"""
        event_type = event.get("type")
        event_data = event.get("data", {})
        
        if event_type in self._handlers:
            for handler in self._handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event_data)
                    else:
                        handler(event_data)
                except Exception as e:
                    logger.error(f"Error handling event {event_type}: {e}")


# Importar asyncio para verificar coroutines
import asyncio
import time















