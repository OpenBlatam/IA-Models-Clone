"""
Document Webhooks - Sistema de Webhooks
========================================

Sistema de webhooks para notificaciones de eventos.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
import aiohttp
import json

logger = logging.getLogger(__name__)


@dataclass
class WebhookEvent:
    """Evento de webhook."""
    event_type: str  # 'analysis_complete', 'quality_check', 'version_added', etc.
    document_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class WebhookConfig:
    """Configuración de webhook."""
    url: str
    secret: Optional[str] = None
    events: List[str] = field(default_factory=lambda: ["*"])  # ["*"] = todos
    timeout: int = 30
    retry_count: int = 3
    enabled: bool = True


class WebhookManager:
    """Gestor de webhooks."""
    
    def __init__(self):
        """Inicializar gestor."""
        self.webhooks: List[WebhookConfig] = []
        self.event_history: List[WebhookEvent] = []
        self.session: Optional[aiohttp.ClientSession] = None
    
    def register_webhook(self, config: WebhookConfig):
        """Registrar webhook."""
        self.webhooks.append(config)
        logger.info(f"Webhook registrado: {config.url}")
    
    def unregister_webhook(self, url: str):
        """Desregistrar webhook."""
        self.webhooks = [w for w in self.webhooks if w.url != url]
        logger.info(f"Webhook desregistrado: {url}")
    
    async def trigger_event(
        self,
        event_type: str,
        document_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None
    ):
        """
        Disparar evento a todos los webhooks registrados.
        
        Args:
            event_type: Tipo de evento
            document_id: ID del documento
            data: Datos adicionales
        """
        event = WebhookEvent(
            event_type=event_type,
            document_id=document_id,
            data=data or {},
            timestamp=datetime.now()
        )
        
        self.event_history.append(event)
        
        # Filtrar webhooks que escuchan este evento
        matching_webhooks = [
            w for w in self.webhooks
            if w.enabled and (event_type in w.events or "*" in w.events)
        ]
        
        # Enviar a todos los webhooks relevantes
        tasks = [
            self._send_webhook(webhook, event)
            for webhook in matching_webhooks
        ]
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_webhook(
        self,
        webhook: WebhookConfig,
        event: WebhookEvent
    ):
        """Enviar webhook."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        payload = {
            "event_type": event.event_type,
            "document_id": event.document_id,
            "data": event.data,
            "timestamp": event.timestamp.isoformat()
        }
        
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "DocumentAnalyzer-Webhook/1.0"
        }
        
        if webhook.secret:
            headers["X-Webhook-Secret"] = webhook.secret
        
        for attempt in range(webhook.retry_count):
            try:
                async with self.session.post(
                    webhook.url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=webhook.timeout)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Webhook enviado exitosamente a {webhook.url}")
                        return
                    else:
                        logger.warning(
                            f"Webhook {webhook.url} respondió con status {response.status}"
                        )
            except Exception as e:
                logger.error(f"Error enviando webhook (intento {attempt + 1}): {e}")
                if attempt < webhook.retry_count - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        logger.error(f"Fallo al enviar webhook a {webhook.url} después de {webhook.retry_count} intentos")
    
    async def close(self):
        """Cerrar sesión."""
        if self.session:
            await self.session.close()
            self.session = None
    
    def get_event_history(
        self,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[WebhookEvent]:
        """Obtener historial de eventos."""
        events = self.event_history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]
    
    def get_webhook_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de webhooks."""
        return {
            "total_webhooks": len(self.webhooks),
            "active_webhooks": len([w for w in self.webhooks if w.enabled]),
            "total_events": len(self.event_history),
            "events_by_type": {
                event_type: len([e for e in self.event_history if e.event_type == event_type])
                for event_type in set(e.event_type for e in self.event_history)
            }
        }


# Eventos predefinidos
class DocumentEvents:
    """Eventos predefinidos."""
    ANALYSIS_COMPLETE = "analysis_complete"
    QUALITY_CHECK = "quality_check"
    GRAMMAR_CHECK = "grammar_check"
    VERSION_ADDED = "version_added"
    VERSION_COMPARED = "version_compared"
    COLLABORATION_ANALYZED = "collaboration_analyzed"
    RECOMMENDATIONS_GENERATED = "recommendations_generated"
    BATCH_COMPLETE = "batch_complete"
    ERROR = "error"


__all__ = [
    "WebhookManager",
    "WebhookConfig",
    "WebhookEvent",
    "DocumentEvents"
]
















