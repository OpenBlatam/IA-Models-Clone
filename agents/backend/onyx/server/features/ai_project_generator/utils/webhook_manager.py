"""
Webhook Manager - Gestor de Webhooks
=====================================

Gestiona webhooks para notificaciones de eventos.
"""

import logging
import httpx
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class WebhookManager:
    """Gestor de webhooks"""

    def __init__(self):
        """Inicializa el gestor de webhooks"""
        self.webhooks: List[Dict[str, Any]] = []

    def register_webhook(
        self,
        url: str,
        events: List[str],
        secret: Optional[str] = None,
    ) -> str:
        """
        Registra un webhook.

        Args:
            url: URL del webhook
            events: Lista de eventos a escuchar
            secret: Secret para verificación (opcional)

        Returns:
            ID del webhook registrado
        """
        webhook_id = f"wh_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        webhook = {
            "id": webhook_id,
            "url": url,
            "events": events,
            "secret": secret,
            "created_at": datetime.now().isoformat(),
            "active": True,
        }

        self.webhooks.append(webhook)
        logger.info(f"Webhook registrado: {webhook_id} para eventos {events}")

        return webhook_id

    async def trigger_webhook(
        self,
        event: str,
        data: Dict[str, Any],
    ):
        """
        Dispara un webhook para un evento.

        Args:
            event: Nombre del evento
            data: Datos del evento
        """
        for webhook in self.webhooks:
            if not webhook.get("active", True):
                continue

            if event not in webhook.get("events", []):
                continue

            try:
                payload = {
                    "event": event,
                    "timestamp": datetime.now().isoformat(),
                    "data": data,
                }

                # Agregar signature si hay secret
                if webhook.get("secret"):
                    import hmac
                    import hashlib
                    signature = hmac.new(
                        webhook["secret"].encode(),
                        json.dumps(payload).encode(),
                        hashlib.sha256
                    ).hexdigest()
                    payload["signature"] = signature

                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        webhook["url"],
                        json=payload,
                        headers={
                            "Content-Type": "application/json",
                            "X-Webhook-Event": event,
                        }
                    )
                    response.raise_for_status()

                logger.info(f"Webhook disparado exitosamente: {webhook['id']}")

            except Exception as e:
                logger.error(f"Error disparando webhook {webhook['id']}: {e}")

    def unregister_webhook(self, webhook_id: str) -> bool:
        """
        Desregistra un webhook.

        Args:
            webhook_id: ID del webhook

        Returns:
            True si se desregistró exitosamente
        """
        for i, webhook in enumerate(self.webhooks):
            if webhook["id"] == webhook_id:
                del self.webhooks[i]
                logger.info(f"Webhook desregistrado: {webhook_id}")
                return True
        return False

    def list_webhooks(self) -> List[Dict[str, Any]]:
        """Lista todos los webhooks registrados"""
        return [
            {
                "id": wh["id"],
                "url": wh["url"],
                "events": wh["events"],
                "active": wh.get("active", True),
                "created_at": wh.get("created_at"),
            }
            for wh in self.webhooks
        ]


