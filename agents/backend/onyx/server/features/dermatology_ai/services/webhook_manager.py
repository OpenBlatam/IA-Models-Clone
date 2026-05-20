"""
Sistema de webhooks para notificaciones
"""

import json
import asyncio
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import aiohttp
from urllib.parse import urlparse


class WebhookEvent(str, Enum):
    """Tipos de eventos de webhook"""
    ANALYSIS_COMPLETED = "analysis.completed"
    ALERT_CREATED = "alert.created"
    PROGRESS_MILESTONE = "progress.milestone"
    CONDITION_DETECTED = "condition.detected"
    RECOMMENDATION_UPDATED = "recommendation.updated"


@dataclass
class Webhook:
    """Configuración de webhook"""
    id: str
    url: str
    events: List[WebhookEvent]
    secret: Optional[str] = None
    active: bool = True
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return asdict(self)


class WebhookManager:
    """Gestor de webhooks"""
    
    def __init__(self):
        """Inicializa el gestor de webhooks"""
        self.webhooks: Dict[str, Webhook] = {}
        self.event_history: List[Dict] = []
    
    def register_webhook(self, url: str, events: List[WebhookEvent],
                        secret: Optional[str] = None) -> str:
        """
        Registra un nuevo webhook
        
        Args:
            url: URL del webhook
            events: Lista de eventos a suscribir
            secret: Secreto para validación (opcional)
            
        Returns:
            ID del webhook
        """
        import hashlib
        webhook_id = hashlib.md5(f"{url}{datetime.now().isoformat()}".encode()).hexdigest()
        
        webhook = Webhook(
            id=webhook_id,
            url=url,
            events=events,
            secret=secret,
            active=True
        )
        
        self.webhooks[webhook_id] = webhook
        return webhook_id
    
    def unregister_webhook(self, webhook_id: str) -> bool:
        """
        Elimina un webhook
        
        Args:
            webhook_id: ID del webhook
            
        Returns:
            True si se eliminó correctamente
        """
        if webhook_id in self.webhooks:
            del self.webhooks[webhook_id]
            return True
        return False
    
    async def trigger_webhook(self, event: WebhookEvent, data: Dict):
        """
        Dispara un webhook para un evento
        
        Args:
            event: Tipo de evento
            data: Datos del evento
        """
        # Encontrar webhooks suscritos al evento
        relevant_webhooks = [
            webhook for webhook in self.webhooks.values()
            if webhook.active and event in webhook.events
        ]
        
        if not relevant_webhooks:
            return
        
        # Preparar payload
        payload = {
            "event": event.value,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        # Enviar a todos los webhooks relevantes
        tasks = []
        for webhook in relevant_webhooks:
            tasks.append(self._send_webhook(webhook, payload))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            self._log_webhook_results(event, relevant_webhooks, results)
    
    async def _send_webhook(self, webhook: Webhook, payload: Dict) -> Dict:
        """
        Envía un webhook individual
        
        Args:
            webhook: Configuración del webhook
            payload: Datos a enviar
            
        Returns:
            Resultado del envío
        """
        try:
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "DermatologyAI-Webhook/1.4.0"
            }
            
            # Agregar firma si hay secreto
            if webhook.secret:
                import hmac
                import hashlib
                signature = hmac.new(
                    webhook.secret.encode(),
                    json.dumps(payload).encode(),
                    hashlib.sha256
                ).hexdigest()
                headers["X-Webhook-Signature"] = signature
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook.url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    result = {
                        "webhook_id": webhook.id,
                        "url": webhook.url,
                        "status_code": response.status,
                        "success": 200 <= response.status < 300,
                        "timestamp": datetime.now().isoformat()
                    }
                    return result
        
        except Exception as e:
            return {
                "webhook_id": webhook.id,
                "url": webhook.url,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _log_webhook_results(self, event: WebhookEvent, webhooks: List[Webhook],
                            results: List):
        """Registra resultados de webhooks"""
        for webhook, result in zip(webhooks, results):
            self.event_history.append({
                "event": event.value,
                "webhook_id": webhook.id,
                "result": result if isinstance(result, dict) else {"error": str(result)},
                "timestamp": datetime.now().isoformat()
            })
        
        # Mantener solo últimos 1000 eventos
        if len(self.event_history) > 1000:
            self.event_history = self.event_history[-1000:]
    
    def get_webhook(self, webhook_id: str) -> Optional[Webhook]:
        """Obtiene un webhook por ID"""
        return self.webhooks.get(webhook_id)
    
    def list_webhooks(self) -> List[Webhook]:
        """Lista todos los webhooks"""
        return list(self.webhooks.values())
    
    def get_webhook_history(self, webhook_id: Optional[str] = None,
                           limit: int = 100) -> List[Dict]:
        """
        Obtiene historial de webhooks
        
        Args:
            webhook_id: ID del webhook (opcional, para filtrar)
            limit: Límite de resultados
            
        Returns:
            Lista de eventos
        """
        history = self.event_history
        
        if webhook_id:
            history = [h for h in history if h.get("webhook_id") == webhook_id]
        
        return history[-limit:]






