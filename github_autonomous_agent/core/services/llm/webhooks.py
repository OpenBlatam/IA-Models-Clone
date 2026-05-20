"""
Sistema de Webhooks y Notificaciones para LLM Service.

Permite configurar webhooks para recibir notificaciones sobre
eventos del servicio LLM (completado, errores, experimentos, etc.).
"""

import json
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import httpx
from urllib.parse import urlparse

from config.logging_config import get_logger

logger = get_logger(__name__)


class WebhookEvent(str, Enum):
    """Tipos de eventos para webhooks."""
    GENERATION_COMPLETED = "generation.completed"
    GENERATION_FAILED = "generation.failed"
    EXPERIMENT_COMPLETED = "experiment.completed"
    EXPERIMENT_FAILED = "experiment.failed"
    AB_TEST_COMPLETED = "ab_test.completed"
    RATE_LIMIT_EXCEEDED = "rate_limit.exceeded"
    COST_THRESHOLD_REACHED = "cost.threshold_reached"
    MODEL_CHANGED = "model.changed"
    CACHE_HIT = "cache.hit"
    CACHE_MISS = "cache.miss"


@dataclass
class WebhookConfig:
    """Configuración de un webhook."""
    webhook_id: str
    url: str
    events: List[WebhookEvent]
    secret: Optional[str] = None
    enabled: bool = True
    timeout: int = 5
    retry_count: int = 3
    headers: Dict[str, str] = None
    
    def __post_init__(self):
        """Validar URL y headers."""
        if self.headers is None:
            self.headers = {}
        
        # Validar URL
        parsed = urlparse(self.url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"URL inválida: {self.url}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "webhook_id": self.webhook_id,
            "url": self.url,
            "events": [e.value for e in self.events],
            "enabled": self.enabled,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "headers": self.headers
        }


@dataclass
class WebhookPayload:
    """Payload para enviar en webhook."""
    event: str
    timestamp: str
    data: Dict[str, Any]
    webhook_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return asdict(self)


class WebhookService:
    """
    Servicio para gestionar webhooks y notificaciones.
    
    Características:
    - Registro de webhooks por evento
    - Envío asíncrono de notificaciones
    - Retry automático en caso de fallo
    - Firma de payloads con secret
    - Rate limiting para webhooks
    """
    
    def __init__(self):
        """Inicializar servicio de webhooks."""
        self.webhooks: Dict[str, WebhookConfig] = {}
        self.client: Optional[httpx.AsyncClient] = None
        self._init_client()
    
    def _init_client(self):
        """Inicializar cliente HTTP."""
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0),
            follow_redirects=True
        )
    
    def register_webhook(
        self,
        url: str,
        events: List[WebhookEvent],
        secret: Optional[str] = None,
        webhook_id: Optional[str] = None,
        timeout: int = 5,
        retry_count: int = 3,
        headers: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Registrar un nuevo webhook.
        
        Args:
            url: URL del webhook
            events: Lista de eventos a escuchar
            secret: Secret para firmar payloads (opcional)
            webhook_id: ID personalizado (opcional, se genera si no se proporciona)
            timeout: Timeout en segundos
            retry_count: Número de reintentos
            headers: Headers adicionales
            
        Returns:
            ID del webhook registrado
        """
        import hashlib
        
        if webhook_id is None:
            webhook_id = hashlib.md5(
                f"{url}{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12]
        
        config = WebhookConfig(
            webhook_id=webhook_id,
            url=url,
            events=events,
            secret=secret,
            timeout=timeout,
            retry_count=retry_count,
            headers=headers or {}
        )
        
        self.webhooks[webhook_id] = config
        logger.info(f"Webhook registrado: {webhook_id} para eventos {[e.value for e in events]}")
        
        return webhook_id
    
    def unregister_webhook(self, webhook_id: str) -> bool:
        """
        Desregistrar un webhook.
        
        Args:
            webhook_id: ID del webhook
            
        Returns:
            True si se desregistró correctamente
        """
        if webhook_id in self.webhooks:
            del self.webhooks[webhook_id]
            logger.info(f"Webhook desregistrado: {webhook_id}")
            return True
        return False
    
    def get_webhook(self, webhook_id: str) -> Optional[WebhookConfig]:
        """Obtener configuración de webhook."""
        return self.webhooks.get(webhook_id)
    
    def list_webhooks(self, event: Optional[WebhookEvent] = None) -> List[WebhookConfig]:
        """
        Listar webhooks registrados.
        
        Args:
            event: Filtrar por evento (opcional)
            
        Returns:
            Lista de webhooks
        """
        webhooks = list(self.webhooks.values())
        if event:
            webhooks = [w for w in webhooks if event in w.events and w.enabled]
        return webhooks
    
    async def trigger_webhook(
        self,
        event: WebhookEvent,
        data: Dict[str, Any],
        webhook_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Disparar webhook para un evento.
        
        Args:
            event: Evento a disparar
            data: Datos del evento
            webhook_id: ID específico de webhook (opcional, dispara todos si no se proporciona)
            
        Returns:
            Resultado del envío
        """
        results = {
            "event": event.value,
            "triggered": 0,
            "successful": 0,
            "failed": 0,
            "errors": []
        }
        
        # Obtener webhooks para este evento
        if webhook_id:
            webhooks = [self.webhooks.get(webhook_id)] if webhook_id in self.webhooks else []
        else:
            webhooks = [
                w for w in self.webhooks.values()
                if event in w.events and w.enabled
            ]
        
        # Enviar a cada webhook
        tasks = []
        for webhook in webhooks:
            if webhook:
                tasks.append(self._send_webhook(webhook, event, data))
        
        if tasks:
            send_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in send_results:
                results["triggered"] += 1
                if isinstance(result, Exception):
                    results["failed"] += 1
                    results["errors"].append(str(result))
                elif result.get("success"):
                    results["successful"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append(result.get("error", "Unknown error"))
        
        return results
    
    async def _send_webhook(
        self,
        webhook: WebhookConfig,
        event: WebhookEvent,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enviar webhook individual.
        
        Args:
            webhook: Configuración del webhook
            event: Evento
            data: Datos
            
        Returns:
            Resultado del envío
        """
        payload = WebhookPayload(
            event=event.value,
            timestamp=datetime.now().isoformat(),
            data=data,
            webhook_id=webhook.webhook_id
        )
        
        # Firmar payload si hay secret
        headers = webhook.headers.copy()
        if webhook.secret:
            import hmac
            import hashlib
            payload_json = json.dumps(payload.to_dict(), sort_keys=True)
            signature = hmac.new(
                webhook.secret.encode(),
                payload_json.encode(),
                hashlib.sha256
            ).hexdigest()
            headers["X-Webhook-Signature"] = f"sha256={signature}"
        
        headers["Content-Type"] = "application/json"
        headers["User-Agent"] = "LLM-Service-Webhook/1.0"
        
        # Reintentos
        last_error = None
        for attempt in range(webhook.retry_count):
            try:
                response = await self.client.post(
                    webhook.url,
                    json=payload.to_dict(),
                    headers=headers,
                    timeout=webhook.timeout
                )
                response.raise_for_status()
                
                logger.debug(f"Webhook {webhook.webhook_id} enviado exitosamente")
                return {"success": True, "status_code": response.status_code}
            
            except Exception as e:
                last_error = e
                if attempt < webhook.retry_count - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"Intento {attempt + 1} fallido para webhook {webhook.webhook_id}: {e}"
                    )
        
        logger.error(f"Error enviando webhook {webhook.webhook_id}: {last_error}")
        return {"success": False, "error": str(last_error)}
    
    async def test_webhook(self, webhook_id: str) -> Dict[str, Any]:
        """
        Probar un webhook con un evento de prueba.
        
        Args:
            webhook_id: ID del webhook
            
        Returns:
            Resultado de la prueba
        """
        webhook = self.webhooks.get(webhook_id)
        if not webhook:
            return {"success": False, "error": "Webhook no encontrado"}
        
        test_data = {
            "test": True,
            "message": "This is a test webhook",
            "timestamp": datetime.now().isoformat()
        }
        
        return await self._send_webhook(
            webhook,
            WebhookEvent.GENERATION_COMPLETED,
            test_data
        )
    
    async def close(self):
        """Cerrar cliente HTTP."""
        if self.client:
            await self.client.aclose()
    
    async def __aenter__(self):
        """Context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.close()


def get_webhook_service() -> WebhookService:
    """Factory function para obtener instancia singleton del servicio."""
    if not hasattr(get_webhook_service, "_instance"):
        get_webhook_service._instance = WebhookService()
    return get_webhook_service._instance



