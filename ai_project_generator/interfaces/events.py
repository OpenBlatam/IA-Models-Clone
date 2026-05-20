"""
Event Interfaces - Interfaces para eventos
==========================================

Define contratos para publicación y suscripción de eventos.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Callable


class IEventPublisher(ABC):
    """Interfaz para publicador de eventos"""
    
    @abstractmethod
    async def publish(
        self,
        event_type: str,
        event_data: Dict[str, Any]
    ) -> bool:
        """Publica un evento"""
        pass
    
    @abstractmethod
    async def publish_batch(
        self,
        events: list
    ) -> Dict[str, bool]:
        """Publica múltiples eventos"""
        pass


class IEventSubscriber(ABC):
    """Interfaz para suscriptor de eventos"""
    
    @abstractmethod
    def subscribe(
        self,
        event_type: str,
        handler: Callable[[Dict[str, Any]], None]
    ):
        """Suscribe un handler a un tipo de evento"""
        pass
    
    @abstractmethod
    async def start_listening(self):
        """Inicia escucha de eventos"""
        pass















