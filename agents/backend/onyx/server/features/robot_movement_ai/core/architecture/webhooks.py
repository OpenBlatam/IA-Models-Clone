"""
Sistema de webhooks para Robot Movement AI v2.0
Notificaciones HTTP a endpoints externos
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class WebhookEvent(str, Enum):
    """Tipos de eventos para webhooks"""
    ROBOT_CONNECTED = "robot.connected"
    ROBOT_DISCONNECTED = "robot.disconnected"
    MOVEMENT_STARTED = "movement.started"
    MOVEMENT_COMPLETED = "movement.completed"
    MOVEMENT_FAILED = "movement.failed"
    ERROR_OCCURRED = "error.occurred"


@dataclass
class Webhook:
    """Configuración de webhook"""
    id: str
    url: str
    events: List[WebhookEvent]
    secret: Optional[str] = None
    timeout: int = 10
    retry_count: int = 3
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)


class WebhookManager:
    """Gestor de webhooks"""
    
    def __init__(self):
        """Inicializar gestor de webhooks"""
        self.webhooks: Dict[str, Webhook] = {}
        self.client: Optional[httpx.AsyncClient] = None
        if HTTPX_AVAILABLE:
            self.client = httpx.AsyncClient(timeout=10.0)
    
    def register(self, webhook: Webhook):
        """Registrar webhook"""
        self.webhooks[webhook.id] = webhook
    
    def unregister(self, webhook_id: str):
        """Desregistrar webhook"""
        self.webhooks.pop(webhook_id, None)
    
    def get_webhooks_for_event(self, event: WebhookEvent) -> List[Webhook]:
        """Obtener webhooks para un evento"""
        return [
            webhook for webhook in self.webhooks.values()
            if webhook.enabled and event in webhook.events
        ]
    
    async def trigger(self, event: WebhookEvent, data: Dict[str, Any]):
        """
        Disparar webhooks para un evento
        
        Args:
            event: Tipo de evento
            data: Datos del evento
        """
        if not HTTPX_AVAILABLE:
            print("Warning: httpx not available, webhooks disabled")
            return
        
        webhooks = self.get_webhooks_for_event(event)
        
        if not webhooks:
            return
        
        # Disparar webhooks en paralelo
        tasks = [self._send_webhook(webhook, event, data) for webhook in webhooks]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_webhook(
        self,
        webhook: Webhook,
        event: WebhookEvent,
        data: Dict[str, Any]
    ):
        """Enviar webhook individual"""
        payload = {
            "event": event.value,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Robot-Movement-AI/2.0"
        }
        
        # Agregar firma si hay secret
        if webhook.secret:
            import hmac
            import hashlib
            signature = hmac.new(
                webhook.secret.encode(),
                json.dumps(payload).encode(),
                hashlib.sha256
            ).hexdigest()
            headers["X-Webhook-Signature"] = signature
        
        # Reintentos
        last_error = None
        for attempt in range(webhook.retry_count):
            try:
                response = await self.client.post(
                    webhook.url,
                    json=payload,
                    headers=headers,
                    timeout=webhook.timeout
                )
                response.raise_for_status()
                return  # Éxito
                
            except Exception as e:
                last_error = e
                if attempt < webhook.retry_count - 1:
                    await asyncio.sleep(2 ** attempt)  # Backoff exponencial
        
        # Todos los intentos fallaron
        print(f"Failed to send webhook {webhook.id} after {webhook.retry_count} attempts: {last_error}")
    
    async def close(self):
        """Cerrar cliente HTTP"""
        if self.client:
            await self.client.aclose()


# Instancia global
_webhook_manager: Optional[WebhookManager] = None


def get_webhook_manager() -> WebhookManager:
    """Obtener instancia global del gestor de webhooks"""
    global _webhook_manager
    if _webhook_manager is None:
        _webhook_manager = WebhookManager()
    return _webhook_manager


async def trigger_webhook(event: WebhookEvent, data: Dict[str, Any]):
    """Helper para disparar webhook"""
    manager = get_webhook_manager()
    await manager.trigger(event, data)




