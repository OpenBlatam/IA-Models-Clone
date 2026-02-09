"""
Servicio de webhooks para notificaciones
"""

import logging
import httpx
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Webhook:
    """Configuración de webhook"""
    url: str
    events: List[str]  # identity_created, content_generated, etc.
    secret: Optional[str] = None
    enabled: bool = True


class WebhookService:
    """Servicio para enviar webhooks"""
    
    def __init__(self):
        self.webhooks: List[Webhook] = []
        self.client = httpx.AsyncClient(timeout=10.0)
    
    def register_webhook(self, webhook: Webhook):
        """Registra un webhook"""
        self.webhooks.append(webhook)
        logger.info(f"Webhook registrado: {webhook.url} para eventos: {webhook.events}")
    
    async def send_webhook(self, event: str, data: Dict[str, Any]):
        """
        Envía webhook para un evento
        
        Args:
            event: Tipo de evento
            data: Datos del evento
        """
        # Filtrar webhooks que escuchan este evento
        relevant_webhooks = [
            wh for wh in self.webhooks 
            if wh.enabled and event in wh.events
        ]
        
        if not relevant_webhooks:
            return
        
        # Preparar payload
        payload = {
            "event": event,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
        
        # Enviar a todos los webhooks relevantes
        tasks = [
            self._send_to_webhook(webhook, payload)
            for webhook in relevant_webhooks
        ]
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_to_webhook(self, webhook: Webhook, payload: Dict[str, Any]):
        """Envía payload a un webhook específico"""
        try:
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "SocialMediaIdentityCloneAI/1.0"
            }
            
            # Agregar firma si hay secret
            if webhook.secret:
                import hmac
                import hashlib
                import json
                payload_str = json.dumps(payload, sort_keys=True)
                signature = hmac.new(
                    webhook.secret.encode(),
                    payload_str.encode(),
                    hashlib.sha256
                ).hexdigest()
                headers["X-Webhook-Signature"] = f"sha256={signature}"
            
            response = await self.client.post(
                webhook.url,
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            
            logger.info(f"Webhook enviado exitosamente a {webhook.url}")
            
        except httpx.HTTPError as e:
            logger.error(f"Error enviando webhook a {webhook.url}: {e}")
        except Exception as e:
            logger.error(f"Error inesperado enviando webhook: {e}", exc_info=True)
    
    async def close(self):
        """Cierra el cliente HTTP"""
        await self.client.aclose()


# Singleton global
_webhook_service: Optional[WebhookService] = None


def get_webhook_service() -> WebhookService:
    """Obtiene instancia singleton del servicio de webhooks"""
    global _webhook_service
    if _webhook_service is None:
        _webhook_service = WebhookService()
    return _webhook_service




