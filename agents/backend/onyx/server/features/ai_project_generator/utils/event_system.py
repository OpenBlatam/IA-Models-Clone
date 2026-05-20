"""
Event System - Sistema de Eventos
==================================

Sistema de eventos para comunicación entre componentes.
"""

import logging
import asyncio
from typing import Dict, Any, List, Callable, Optional
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class EventSystem:
    """Sistema de eventos"""

    def __init__(self):
        """Inicializa el sistema de eventos"""
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_history: List[Dict[str, Any]] = []

    def subscribe(
        self,
        event_type: str,
        handler: Callable,
    ):
        """
        Suscribe un handler a un tipo de evento.

        Args:
            event_type: Tipo de evento
            handler: Función handler
        """
        self.subscribers[event_type].append(handler)
        logger.info(f"Handler suscrito a evento: {event_type}")

    async def emit(
        self,
        event_type: str,
        data: Dict[str, Any],
    ):
        """
        Emite un evento.

        Args:
            event_type: Tipo de evento
            data: Datos del evento
        """
        event = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        }

        # Guardar en historial
        self.event_history.append(event)
        if len(self.event_history) > 1000:  # Limitar historial
            self.event_history = self.event_history[-1000:]

        # Notificar a suscriptores
        if event_type in self.subscribers:
            tasks = []
            for handler in self.subscribers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        tasks.append(handler(event))
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Error en handler de evento {event_type}: {e}")

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        logger.debug(f"Evento emitido: {event_type}")

    def get_event_history(
        self,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Obtiene el historial de eventos.

        Args:
            event_type: Filtrar por tipo (opcional)
            limit: Límite de resultados

        Returns:
            Lista de eventos
        """
        events = self.event_history

        if event_type:
            events = [e for e in events if e["type"] == event_type]

        return events[-limit:]

    def get_event_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de eventos"""
        event_counts = defaultdict(int)
        for event in self.event_history:
            event_counts[event["type"]] += 1

        return {
            "total_events": len(self.event_history),
            "events_by_type": dict(event_counts),
            "subscribers_by_type": {
                event_type: len(handlers)
                for event_type, handlers in self.subscribers.items()
            },
        }


