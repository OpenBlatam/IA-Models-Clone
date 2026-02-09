"""
Webhook Manager - Sistema de webhooks y notificaciones
=====================================================
"""

import logging
import httpx
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class WebhookManager:
    """
    Gestiona webhooks y notificaciones para eventos del sistema.
    """
    
    def __init__(self):
        """Inicializar gestor de webhooks"""
        self.webhooks: Dict[str, List[Dict[str, Any]]] = {
            "paper_uploaded": [],
            "model_trained": [],
            "code_improved": [],
            "batch_completed": [],
            "error_occurred": []
        }
        self.client = httpx.AsyncClient(timeout=10.0)
    
    def register_webhook(
        self,
        event_type: str,
        url: str,
        secret: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Registra un webhook para un tipo de evento.
        
        Args:
            event_type: Tipo de evento
            url: URL del webhook
            secret: Secret para validación (opcional)
            headers: Headers adicionales (opcional)
            
        Returns:
            ID del webhook registrado
        """
        import uuid
        
        webhook_id = str(uuid.uuid4())
        
        webhook_config = {
            "id": webhook_id,
            "url": url,
            "secret": secret,
            "headers": headers or {},
            "created_at": datetime.now().isoformat(),
            "active": True
        }
        
        if event_type in self.webhooks:
            self.webhooks[event_type].append(webhook_config)
            logger.info(f"Webhook registrado: {event_type} -> {url}")
        else:
            logger.warning(f"Tipo de evento desconocido: {event_type}")
        
        return webhook_id
    
    async def trigger_webhook(
        self,
        event_type: str,
        payload: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Dispara webhooks para un evento.
        
        Args:
            event_type: Tipo de evento
            payload: Datos del evento
            
        Returns:
            Resultados de los webhooks
        """
        if event_type not in self.webhooks:
            logger.warning(f"Tipo de evento desconocido: {event_type}")
            return []
        
        results = []
        webhooks = [w for w in self.webhooks[event_type] if w.get("active", True)]
        
        if not webhooks:
            return results
        
        # Preparar payload
        webhook_payload = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": payload
        }
        
        # Disparar webhooks en paralelo
        tasks = [
            self._send_webhook(webhook, webhook_payload)
            for webhook in webhooks
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Procesar resultados
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "webhook_id": webhooks[i]["id"],
                    "success": False,
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _send_webhook(
        self,
        webhook: Dict[str, Any],
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Envía un webhook individual"""
        try:
            headers = webhook.get("headers", {}).copy()
            headers["Content-Type"] = "application/json"
            
            # Agregar signature si hay secret
            if webhook.get("secret"):
                import hmac
                import hashlib
                payload_str = json.dumps(payload, sort_keys=True)
                signature = hmac.new(
                    webhook["secret"].encode(),
                    payload_str.encode(),
                    hashlib.sha256
                ).hexdigest()
                headers["X-Webhook-Signature"] = f"sha256={signature}"
            
            response = await self.client.post(
                webhook["url"],
                json=payload,
                headers=headers
            )
            
            response.raise_for_status()
            
            return {
                "webhook_id": webhook["id"],
                "success": True,
                "status_code": response.status_code
            }
            
        except Exception as e:
            logger.error(f"Error enviando webhook {webhook['id']}: {e}")
            return {
                "webhook_id": webhook["id"],
                "success": False,
                "error": str(e)
            }
    
    def unregister_webhook(self, event_type: str, webhook_id: str) -> bool:
        """
        Desregistra un webhook.
        
        Args:
            event_type: Tipo de evento
            webhook_id: ID del webhook
            
        Returns:
            True si se desregistró exitosamente
        """
        if event_type not in self.webhooks:
            return False
        
        original_count = len(self.webhooks[event_type])
        self.webhooks[event_type] = [
            w for w in self.webhooks[event_type]
            if w["id"] != webhook_id
        ]
        
        removed = len(self.webhooks[event_type]) < original_count
        
        if removed:
            logger.info(f"Webhook desregistrado: {webhook_id}")
        
        return removed
    
    def list_webhooks(self, event_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Lista webhooks registrados.
        
        Args:
            event_type: Tipo de evento (opcional)
            
        Returns:
            Lista de webhooks
        """
        if event_type:
            return {
                event_type: [
                    {
                        "id": w["id"],
                        "url": w["url"],
                        "created_at": w["created_at"],
                        "active": w.get("active", True)
                    }
                    for w in self.webhooks.get(event_type, [])
                ]
            }
        else:
            return {
                event: [
                    {
                        "id": w["id"],
                        "url": w["url"],
                        "created_at": w["created_at"],
                        "active": w.get("active", True)
                    }
                    for w in webhooks
                ]
                for event, webhooks in self.webhooks.items()
            }
    
    async def close(self):
        """Cierra el cliente HTTP"""
        await self.client.aclose()




