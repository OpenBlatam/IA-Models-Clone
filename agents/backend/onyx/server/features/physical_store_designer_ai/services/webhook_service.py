"""
Webhook Service - Sistema de webhooks
"""

import logging
import httpx
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class WebhookEvent(str, Enum):
    """Eventos de webhook"""
    DESIGN_CREATED = "design.created"
    DESIGN_UPDATED = "design.updated"
    DESIGN_DELETED = "design.deleted"
    ANALYSIS_COMPLETED = "analysis.completed"
    FEEDBACK_ADDED = "feedback.added"
    ALERT_TRIGGERED = "alert.triggered"


class WebhookService:
    """Servicio para webhooks"""
    
    def __init__(self):
        self.webhooks: Dict[str, List[Dict[str, Any]]] = {}
        self.client = httpx.AsyncClient(timeout=10.0)
    
    def register_webhook(
        self,
        user_id: str,
        url: str,
        events: List[WebhookEvent],
        secret: Optional[str] = None
    ) -> Dict[str, Any]:
        """Registrar webhook"""
        
        webhook_id = f"wh_{user_id}_{len(self.webhooks.get(user_id, [])) + 1}"
        
        webhook = {
            "webhook_id": webhook_id,
            "user_id": user_id,
            "url": url,
            "events": [e.value for e in events],
            "secret": secret,
            "created_at": datetime.now().isoformat(),
            "is_active": True,
            "success_count": 0,
            "failure_count": 0
        }
        
        if user_id not in self.webhooks:
            self.webhooks[user_id] = []
        
        self.webhooks[user_id].append(webhook)
        
        logger.info(f"Webhook registrado: {webhook_id} para {url}")
        return webhook
    
    def unregister_webhook(
        self,
        user_id: str,
        webhook_id: str
    ) -> bool:
        """Desregistrar webhook"""
        user_webhooks = self.webhooks.get(user_id, [])
        
        for i, webhook in enumerate(user_webhooks):
            if webhook["webhook_id"] == webhook_id:
                webhook["is_active"] = False
                webhook["unregistered_at"] = datetime.now().isoformat()
                return True
        
        return False
    
    async def trigger_webhook(
        self,
        event: WebhookEvent,
        data: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Disparar webhook para evento"""
        
        triggered = []
        failed = []
        
        # Si hay user_id, solo para ese usuario
        if user_id:
            webhooks_to_check = self.webhooks.get(user_id, [])
        else:
            # Todos los webhooks activos
            webhooks_to_check = []
            for user_webhooks in self.webhooks.values():
                webhooks_to_check.extend(user_webhooks)
        
        for webhook in webhooks_to_check:
            if not webhook.get("is_active", True):
                continue
            
            if event.value not in webhook.get("events", []):
                continue
            
            try:
                payload = {
                    "event": event.value,
                    "timestamp": datetime.now().isoformat(),
                    "data": data
                }
                
                headers = {
                    "Content-Type": "application/json",
                    "User-Agent": "PhysicalStoreDesignerAI/1.0"
                }
                
                if webhook.get("secret"):
                    # En producción, agregar firma HMAC
                    headers["X-Webhook-Signature"] = "signature_placeholder"
                
                response = await self.client.post(
                    webhook["url"],
                    json=payload,
                    headers=headers
                )
                
                if response.status_code in [200, 201, 202]:
                    webhook["success_count"] = webhook.get("success_count", 0) + 1
                    triggered.append(webhook["webhook_id"])
                else:
                    webhook["failure_count"] = webhook.get("failure_count", 0) + 1
                    failed.append({
                        "webhook_id": webhook["webhook_id"],
                        "status_code": response.status_code,
                        "error": response.text[:100]
                    })
                
            except Exception as e:
                logger.error(f"Error disparando webhook {webhook['webhook_id']}: {e}")
                webhook["failure_count"] = webhook.get("failure_count", 0) + 1
                failed.append({
                    "webhook_id": webhook["webhook_id"],
                    "error": str(e)
                })
        
        return {
            "event": event.value,
            "triggered_count": len(triggered),
            "failed_count": len(failed),
            "triggered": triggered,
            "failed": failed
        }
    
    def get_webhooks(self, user_id: str) -> List[Dict[str, Any]]:
        """Obtener webhooks de usuario"""
        return [
            {
                "webhook_id": wh["webhook_id"],
                "url": wh["url"],
                "events": wh["events"],
                "is_active": wh.get("is_active", True),
                "created_at": wh["created_at"],
                "success_count": wh.get("success_count", 0),
                "failure_count": wh.get("failure_count", 0)
            }
            for wh in self.webhooks.get(user_id, [])
        ]
    
    async def close(self):
        """Cerrar cliente HTTP"""
        await self.client.aclose()




