"""
Webhook System - Sistema de webhooks para eventos
==================================================
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from enum import Enum
import json
import aiohttp

logger = logging.getLogger(__name__)


class WebhookEvent(str, Enum):
    """Tipos de eventos para webhooks"""
    PROTOTYPE_GENERATED = "prototype.generated"
    PROTOTYPE_UPDATED = "prototype.updated"
    VALIDATION_COMPLETED = "validation.completed"
    COST_ANALYSIS_COMPLETED = "cost_analysis.completed"
    FEASIBILITY_ANALYZED = "feasibility.analyzed"
    MATERIAL_SEARCHED = "material.searched"
    EXPORT_COMPLETED = "export.completed"


class WebhookSystem:
    """Sistema de webhooks"""
    
    def __init__(self):
        self.webhooks: Dict[str, List[Dict[str, Any]]] = {}
        self.event_history: List[Dict[str, Any]] = []
    
    def register_webhook(self, user_id: str, url: str, events: List[WebhookEvent],
                        secret: Optional[str] = None, active: bool = True) -> str:
        """
        Registra un webhook
        
        Returns:
            Webhook ID
        """
        import uuid
        webhook_id = str(uuid.uuid4())
        
        webhook_config = {
            "id": webhook_id,
            "user_id": user_id,
            "url": url,
            "events": [e.value for e in events],
            "secret": secret,
            "active": active,
            "created_at": datetime.now().isoformat(),
            "last_triggered": None,
            "success_count": 0,
            "failure_count": 0
        }
        
        if user_id not in self.webhooks:
            self.webhooks[user_id] = []
        
        self.webhooks[user_id].append(webhook_config)
        
        logger.info(f"Webhook registrado: {webhook_id} para {user_id}")
        return webhook_id
    
    async def trigger_webhook(self, event: WebhookEvent, data: Dict[str, Any],
                            user_id: Optional[str] = None):
        """Dispara webhooks para un evento"""
        # Registrar evento en historial
        self.event_history.append({
            "event": event.value,
            "data": data,
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id
        })
        
        # Limpiar historial antiguo (últimos 1000 eventos)
        if len(self.event_history) > 1000:
            self.event_history = self.event_history[-1000:]
        
        # Encontrar webhooks relevantes
        webhooks_to_trigger = []
        
        if user_id:
            user_webhooks = self.webhooks.get(user_id, [])
            for webhook in user_webhooks:
                if (webhook["active"] and 
                    event.value in webhook["events"]):
                    webhooks_to_trigger.append(webhook)
        else:
            # Buscar en todos los webhooks
            for user_webhooks in self.webhooks.values():
                for webhook in user_webhooks:
                    if (webhook["active"] and 
                        event.value in webhook["events"]):
                        webhooks_to_trigger.append(webhook)
        
        # Disparar webhooks
        tasks = []
        for webhook in webhooks_to_trigger:
            tasks.append(self._send_webhook(webhook, event, data))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_webhook(self, webhook: Dict[str, Any], event: WebhookEvent,
                          data: Dict[str, Any]):
        """Envía un webhook"""
        payload = {
            "event": event.value,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "3D-Prototype-AI/1.0"
        }
        
        # Agregar firma si hay secret
        if webhook.get("secret"):
            import hmac
            import hashlib
            signature = hmac.new(
                webhook["secret"].encode(),
                json.dumps(payload).encode(),
                hashlib.sha256
            ).hexdigest()
            headers["X-Webhook-Signature"] = f"sha256={signature}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook["url"],
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    webhook["last_triggered"] = datetime.now().isoformat()
                    
                    if response.status == 200:
                        webhook["success_count"] += 1
                        logger.info(f"Webhook {webhook['id']} enviado exitosamente")
                    else:
                        webhook["failure_count"] += 1
                        logger.warning(f"Webhook {webhook['id']} falló: {response.status}")
        
        except Exception as e:
            webhook["failure_count"] += 1
            logger.error(f"Error enviando webhook {webhook['id']}: {e}")
    
    def get_webhooks(self, user_id: str) -> List[Dict[str, Any]]:
        """Obtiene webhooks de un usuario"""
        return self.webhooks.get(user_id, [])
    
    def delete_webhook(self, user_id: str, webhook_id: str) -> bool:
        """Elimina un webhook"""
        user_webhooks = self.webhooks.get(user_id, [])
        original_count = len(user_webhooks)
        self.webhooks[user_id] = [
            w for w in user_webhooks if w["id"] != webhook_id
        ]
        return len(self.webhooks[user_id]) < original_count
    
    def toggle_webhook(self, user_id: str, webhook_id: str, active: bool) -> bool:
        """Activa/desactiva un webhook"""
        user_webhooks = self.webhooks.get(user_id, [])
        for webhook in user_webhooks:
            if webhook["id"] == webhook_id:
                webhook["active"] = active
                return True
        return False
    
    def get_event_history(self, event: Optional[WebhookEvent] = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """Obtiene historial de eventos"""
        history = self.event_history
        
        if event:
            history = [e for e in history if e["event"] == event.value]
        
        return history[-limit:]




