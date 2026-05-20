"""
Webhooks endpoints for event notifications
"""

from fastapi import Query
from typing import List, Optional
import logging

from ..base_router import BaseRouter

logger = logging.getLogger(__name__)


class WebhooksRouter(BaseRouter):
    """Router for webhooks endpoints"""
    
    def __init__(self):
        super().__init__(prefix="/webhooks", tags=["Webhooks"])
        self._register_routes()
    
    def _register_routes(self):
        """Register all webhooks routes"""
        
        @self.router.post("", response_model=dict)
        @self.handle_exceptions
        async def register_webhook(
            url: str = Query(..., description="URL del webhook"),
            events: List[str] = Query(..., description="Eventos a suscribir"),
            user_id: Optional[str] = Query(None)
        ):
            """Registra un nuevo webhook"""
            webhook_service = self.get_service("webhook_service")
            webhook_id = webhook_service.register_webhook(url, events, user_id)
            return self.success_response({
                "webhook_id": webhook_id
            }, message="Webhook registrado")
        
        @self.router.delete("/{webhook_id}", response_model=dict)
        @self.handle_exceptions
        async def delete_webhook(webhook_id: str):
            """Elimina un webhook"""
            webhook_service = self.get_service("webhook_service")
            success = webhook_service.delete_webhook(webhook_id)
            self.require_success(success, "Webhook no encontrado", status_code=404)
            return self.success_response(None, message="Webhook eliminado")
        
        @self.router.get("", response_model=dict)
        @self.handle_exceptions
        async def list_webhooks(user_id: Optional[str] = Query(None)):
            """Lista los webhooks de un usuario"""
            webhook_service = self.get_service("webhook_service")
            webhooks = webhook_service.list_webhooks(user_id)
            return self.list_response(webhooks, key="webhooks")


def get_webhooks_router() -> WebhooksRouter:
    """Factory function to get webhooks router"""
    return WebhooksRouter()

