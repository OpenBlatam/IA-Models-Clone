"""
Servicio de Webhooks para integraciones externas con mejoras.
"""

import asyncio
import hmac
import hashlib
import json
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from enum import Enum
from urllib.parse import urlparse

import httpx
from config.logging_config import get_logger
from config.di_setup import get_service
from core.exceptions import GitHubAgentError

logger = get_logger(__name__)


class WebhookEvent(str, Enum):
    """Tipos de eventos de webhook."""
    TASK_CREATED = "task.created"
    TASK_UPDATED = "task.updated"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    AGENT_STARTED = "agent.started"
    AGENT_STOPPED = "agent.stopped"
    AGENT_PAUSED = "agent.paused"
    AGENT_RESUMED = "agent.resumed"


class Webhook:
    """Representa un webhook."""
    
    def __init__(
        self,
        webhook_id: str,
        url: str,
        events: List[WebhookEvent],
        secret: Optional[str] = None,
        enabled: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializar webhook con validaciones.
        
        Args:
            webhook_id: ID único del webhook (debe ser string no vacío)
            url: URL del webhook (debe ser URL válida)
            events: Eventos a los que suscribirse (debe ser lista no vacía)
            secret: Secret para firma (opcional)
            enabled: Si está habilitado
            metadata: Metadata adicional
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        # Validaciones
        if not webhook_id or not isinstance(webhook_id, str) or not webhook_id.strip():
            raise ValueError("webhook_id debe ser un string no vacío")
        
        if not url or not isinstance(url, str) or not url.strip():
            raise ValueError("url debe ser un string no vacío")
        
        if not events or not isinstance(events, list) or len(events) == 0:
            raise ValueError("events debe ser una lista no vacía de WebhookEvent")
        
        # Validar URL
        parsed = urlparse(url.strip())
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"URL inválida: {url}")
        
        if parsed.scheme not in ["http", "https"]:
            raise ValueError(f"Esquema no soportado: {parsed.scheme}. Solo se permiten http y https")
        
        self.webhook_id = webhook_id.strip()
        self.url = url.strip()
        self.events = events
        self.secret = secret.strip() if secret and isinstance(secret, str) else None
        self.enabled = bool(enabled)
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.stats = {
            "total_attempts": 0,
            "successful": 0,
            "failed": 0,
            "last_attempt": None,
            "last_success": None,
            "last_failure": None
        }
        
        logger.debug(f"Webhook inicializado: {self.webhook_id} -> {self.url} ({len(self.events)} eventos)")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir webhook a diccionario."""
        return {
            "webhook_id": self.webhook_id,
            "url": self.url,
            "events": [e.value for e in self.events],
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "stats": self.stats.copy()
        }


class WebhookService:
    """
    Servicio para gestionar y enviar webhooks con mejoras.
    
    Attributes:
        webhooks: Diccionario de webhooks registrados
        http_client: Cliente HTTP asíncrono
        max_retries: Número máximo de reintentos
        retry_delay: Delay base entre reintentos en segundos
    """
    
    def __init__(
        self,
        timeout: float = 10.0,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Inicializar servicio de webhooks con validaciones.
        
        Args:
            timeout: Timeout para requests HTTP en segundos (default: 10.0)
            max_retries: Número máximo de reintentos (default: 3)
            retry_delay: Delay base entre reintentos en segundos (default: 1.0)
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        # Validaciones
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError(f"timeout debe ser un número positivo, recibido: {timeout}")
        
        if not isinstance(max_retries, int) or max_retries < 1:
            raise ValueError(f"max_retries debe ser un entero positivo, recibido: {max_retries}")
        
        if not isinstance(retry_delay, (int, float)) or retry_delay <= 0:
            raise ValueError(f"retry_delay debe ser un número positivo, recibido: {retry_delay}")
        
        self.webhooks: Dict[str, Webhook] = {}
        self.http_client = httpx.AsyncClient(timeout=timeout)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        logger.info(
            f"WebhookService inicializado: timeout={timeout}s, "
            f"max_retries={max_retries}, retry_delay={retry_delay}s"
        )
    
    def register_webhook(
        self,
        url: str,
        events: List[WebhookEvent],
        secret: Optional[str] = None,
        webhook_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Webhook:
        """
        Registrar nuevo webhook con validaciones mejoradas.
        
        Args:
            url: URL del webhook (debe ser URL válida)
            events: Eventos a los que suscribirse (debe ser lista no vacía)
            secret: Secret para firma (opcional)
            webhook_id: ID del webhook (opcional, se genera si no se proporciona)
            metadata: Metadata adicional
            
        Returns:
            Webhook registrado
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        # Validar URL
        if not url or not isinstance(url, str) or not url.strip():
            raise ValueError("url debe ser un string no vacío")
        
        url = url.strip()
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"URL inválida: {url}")
        
        if parsed.scheme not in ["http", "https"]:
            raise ValueError(f"Esquema no soportado: {parsed.scheme}. Solo se permiten http y https")
        
        # Validar events
        if not events or not isinstance(events, list) or len(events) == 0:
            raise ValueError("events debe ser una lista no vacía de WebhookEvent")
        
        # Validar que todos sean WebhookEvent
        for event in events:
            if not isinstance(event, WebhookEvent):
                raise ValueError(f"Todos los eventos deben ser WebhookEvent, recibido: {type(event)}")
        
        # Generar ID si no se proporciona
        if webhook_id is None:
            import secrets
            webhook_id = f"wh_{secrets.token_urlsafe(16)}"
        elif not isinstance(webhook_id, str) or not webhook_id.strip():
            raise ValueError("webhook_id debe ser un string no vacío si se proporciona")
        else:
            webhook_id = webhook_id.strip()
        
        # Verificar que no exista
        if webhook_id in self.webhooks:
            logger.warning(f"Webhook {webhook_id} ya existe. Sobrescribiendo...")
        
        try:
            webhook = Webhook(
                webhook_id=webhook_id,
                url=url,
                events=events,
                secret=secret,
                metadata=metadata
            )
            
            self.webhooks[webhook_id] = webhook
            logger.info(
                f"✅ Webhook registrado: {webhook_id} -> {url} "
                f"({len(events)} eventos: {', '.join([e.value for e in events])})"
            )
            
            return webhook
        except Exception as e:
            logger.error(f"Error al registrar webhook {webhook_id}: {e}", exc_info=True)
            raise ValueError(f"Error al registrar webhook: {e}") from e
    
    def _sign_payload(self, payload: Dict[str, Any], secret: str) -> str:
        """
        Firmar payload con secret usando HMAC-SHA256.
        
        Args:
            payload: Payload a firmar (debe ser serializable a JSON)
            secret: Secret para firma (debe ser string no vacío)
            
        Returns:
            Firma HMAC en formato hexadecimal
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        # Validaciones
        if not isinstance(payload, dict):
            raise ValueError(f"payload debe ser un diccionario, recibido: {type(payload)}")
        
        if not secret or not isinstance(secret, str) or not secret.strip():
            raise ValueError("secret debe ser un string no vacío")
        
        try:
            payload_str = json.dumps(payload, sort_keys=True, ensure_ascii=False)
            signature = hmac.new(
                secret.encode('utf-8'),
                payload_str.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            logger.debug(f"Payload firmado exitosamente (longitud: {len(payload_str)} caracteres)")
            return signature
        except (TypeError, ValueError) as e:
            logger.error(f"Error al firmar payload: {e}", exc_info=True)
            raise ValueError(f"Error al firmar payload: {e}") from e
    
    async def send_webhook(
        self,
        webhook: Webhook,
        event: WebhookEvent,
        data: Dict[str, Any]
    ) -> bool:
        """
        Enviar webhook con validaciones y mejor manejo de errores.
        
        Args:
            webhook: Webhook a enviar (debe ser instancia de Webhook)
            event: Evento que disparó el webhook (debe ser WebhookEvent)
            data: Datos del evento (debe ser diccionario serializable)
            
        Returns:
            True si se envió exitosamente, False en caso contrario
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        # Validaciones
        if not isinstance(webhook, Webhook):
            raise ValueError(f"webhook debe ser una instancia de Webhook, recibido: {type(webhook)}")
        
        if not isinstance(event, WebhookEvent):
            raise ValueError(f"event debe ser un WebhookEvent, recibido: {type(event)}")
        
        if not isinstance(data, dict):
            raise ValueError(f"data debe ser un diccionario, recibido: {type(data)}")
        
        if not webhook.enabled:
            logger.debug(f"Webhook {webhook.webhook_id} está deshabilitado, omitiendo envío")
            return False
        
        if event not in webhook.events:
            logger.debug(
                f"Webhook {webhook.webhook_id} no está suscrito al evento {event.value}, "
                f"omitindo envío"
            )
            return False
        
        payload = {
            "event": event.value,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "GitHub-Autonomous-Agent/1.0"
        }
        
        # Agregar firma si hay secret
        if webhook.secret:
            signature = self._sign_payload(payload, webhook.secret)
            headers["X-Webhook-Signature"] = f"sha256={signature}"
        
        webhook.stats["total_attempts"] += 1
        webhook.stats["last_attempt"] = datetime.now().isoformat()
        
        # Intentar enviar con retry
        last_error = None
        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    f"Enviando webhook {webhook.webhook_id} (intento {attempt + 1}/{self.max_retries})"
                )
                
                response = await self.http_client.post(
                    webhook.url,
                    json=payload,
                    headers=headers,
                    timeout=self.http_client.timeout
                )
                
                if response.status_code < 400:
                    webhook.stats["successful"] += 1
                    webhook.stats["last_success"] = datetime.now().isoformat()
                    logger.info(
                        f"✅ Webhook {webhook.webhook_id} enviado exitosamente "
                        f"(status: {response.status_code}, evento: {event.value})"
                    )
                    return True
                else:
                    error_msg = (
                        f"Webhook {webhook.webhook_id} falló con status {response.status_code}. "
                        f"Respuesta: {response.text[:200]}"
                    )
                    logger.warning(error_msg)
                    last_error = f"HTTP {response.status_code}: {response.text[:100]}"
                    
            except httpx.TimeoutException as e:
                error_msg = f"Timeout enviando webhook {webhook.webhook_id}: {e}"
                logger.warning(error_msg)
                last_error = f"Timeout: {str(e)}"
            except httpx.RequestError as e:
                error_msg = f"Error de request enviando webhook {webhook.webhook_id}: {e}"
                logger.error(error_msg, exc_info=True)
                last_error = f"Request error: {str(e)}"
            except Exception as e:
                error_msg = f"Error inesperado enviando webhook {webhook.webhook_id}: {e}"
                logger.error(error_msg, exc_info=True)
                last_error = f"Unexpected error: {str(e)}"
            
            # Esperar antes de reintentar (exponential backoff)
            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                logger.debug(f"Esperando {delay}s antes de reintentar...")
                await asyncio.sleep(delay)
        
        # Todos los intentos fallaron
        webhook.stats["failed"] += 1
        webhook.stats["last_failure"] = datetime.now().isoformat()
        logger.error(
            f"❌ Webhook {webhook.webhook_id} falló después de {self.max_retries} intentos. "
            f"Último error: {last_error}"
        )
        return False
    
    async def trigger_event(
        self,
        event: WebhookEvent,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Disparar evento a todos los webhooks suscritos.
        
        Args:
            event: Evento a disparar
            data: Datos del evento
            
        Returns:
            Resultado del envío
        """
        results = {
            "event": event.value,
            "total_webhooks": 0,
            "successful": 0,
            "failed": 0,
            "details": []
        }
        
        for webhook in self.webhooks.values():
            if event not in webhook.events:
                continue
            
            results["total_webhooks"] += 1
            success = await self.send_webhook(webhook, event, data)
            
            if success:
                results["successful"] += 1
            else:
                results["failed"] += 1
            
            results["details"].append({
                "webhook_id": webhook.webhook_id,
                "url": webhook.url,
                "success": success
            })
        
        return results
    
    def get_webhook(self, webhook_id: str) -> Optional[Webhook]:
        """
        Obtener webhook por ID.
        
        Args:
            webhook_id: ID del webhook
            
        Returns:
            Webhook o None
        """
        return self.webhooks.get(webhook_id)
    
    def list_webhooks(self, enabled_only: bool = False) -> List[Dict[str, Any]]:
        """
        Listar webhooks.
        
        Args:
            enabled_only: Solo webhooks habilitados
            
        Returns:
            Lista de webhooks
        """
        webhooks = [
            webhook.to_dict()
            for webhook in self.webhooks.values()
            if not enabled_only or webhook.enabled
        ]
        return webhooks
    
    def delete_webhook(self, webhook_id: str) -> bool:
        """
        Eliminar webhook.
        
        Args:
            webhook_id: ID del webhook
            
        Returns:
            True si se eliminó exitosamente
        """
        if webhook_id in self.webhooks:
            del self.webhooks[webhook_id]
            logger.info(f"Webhook eliminado: {webhook_id}")
            return True
        return False
    
    def enable_webhook(self, webhook_id: str) -> bool:
        """Habilitar webhook."""
        webhook = self.webhooks.get(webhook_id)
        if webhook:
            webhook.enabled = True
            return True
        return False
    
    def disable_webhook(self, webhook_id: str) -> bool:
        """Deshabilitar webhook."""
        webhook = self.webhooks.get(webhook_id)
        if webhook:
            webhook.enabled = False
            return True
        return False
    
    async def close(self) -> None:
        """Cerrar cliente HTTP."""
        await self.http_client.aclose()
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            "total_webhooks": len(self.webhooks),
            "enabled_webhooks": len([w for w in self.webhooks.values() if w.enabled]),
            "webhooks": [
                {
                    "webhook_id": w.webhook_id,
                    "url": w.url,
                    "events_count": len(w.events),
                    "stats": w.stats
                }
                for w in self.webhooks.values()
            ]
        }

