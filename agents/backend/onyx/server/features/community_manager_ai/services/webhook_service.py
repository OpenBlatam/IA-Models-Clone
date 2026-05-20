"""
Webhook Service - Servicio de Webhooks
=======================================

Sistema para manejar webhooks de plataformas sociales.
"""

import logging
import hmac
import hashlib
from typing import Dict, Any, Optional, Callable, List
from collections import defaultdict

logger = logging.getLogger(__name__)


class WebhookService:
    """Servicio para manejar webhooks"""
    
    def __init__(self):
        """Inicializar servicio de webhooks"""
        self.handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.verification_tokens: Dict[str, str] = {}
        logger.info("Webhook Service inicializado")
    
    def register_handler(
        self,
        platform: str,
        event_type: str,
        handler: Callable[[Dict[str, Any]], None]
    ):
        """
        Registrar handler para un evento de webhook
        
        Args:
            platform: Plataforma (facebook, instagram, etc.)
            event_type: Tipo de evento
            handler: Función que maneja el evento
        """
        key = f"{platform}:{event_type}"
        self.handlers[key].append(handler)
        logger.info(f"Handler registrado para {key}")
    
    def handle_webhook(
        self,
        platform: str,
        payload: Dict[str, Any],
        signature: Optional[str] = None
    ) -> bool:
        """
        Manejar un webhook
        
        Args:
            platform: Plataforma
            payload: Datos del webhook
            signature: Firma para verificación (opcional)
            
        Returns:
            True si se manejó exitosamente
        """
        # Verificar firma si está disponible
        if signature and not self.verify_signature(platform, payload, signature):
            logger.warning(f"Firma inválida para webhook de {platform}")
            return False
        
        # Determinar tipo de evento
        event_type = self._extract_event_type(platform, payload)
        
        if not event_type:
            logger.warning(f"No se pudo determinar tipo de evento para {platform}")
            return False
        
        # Ejecutar handlers
        key = f"{platform}:{event_type}"
        handlers = self.handlers.get(key, [])
        
        if not handlers:
            logger.debug(f"No hay handlers para {key}")
            return False
        
        for handler in handlers:
            try:
                handler(payload)
            except Exception as e:
                logger.error(f"Error en handler de webhook: {e}")
        
        logger.info(f"Webhook manejado: {key}")
        return True
    
    def verify_signature(
        self,
        platform: str,
        payload: Dict[str, Any],
        signature: str
    ) -> bool:
        """
        Verificar firma de webhook
        
        Args:
            platform: Plataforma
            payload: Datos del webhook
            signature: Firma recibida
            
        Returns:
            True si la firma es válida
        """
        secret = self.verification_tokens.get(platform)
        if not secret:
            logger.warning(f"No hay secret configurado para {platform}")
            return False
        
        # Verificar según plataforma
        if platform == "facebook":
            return self._verify_facebook_signature(payload, signature, secret)
        elif platform == "instagram":
            return self._verify_instagram_signature(payload, signature, secret)
        # Agregar más plataformas según sea necesario
        
        return True  # Por defecto, permitir si no hay verificación específica
    
    def _verify_facebook_signature(
        self,
        payload: Dict[str, Any],
        signature: str,
        secret: str
    ) -> bool:
        """Verificar firma de Facebook"""
        # Implementar verificación de Facebook
        # Facebook usa HMAC SHA256
        try:
            import json
            payload_str = json.dumps(payload, sort_keys=True)
            expected_signature = hmac.new(
                secret.encode(),
                payload_str.encode(),
                hashlib.sha256
            ).hexdigest()
            return hmac.compare_digest(signature, expected_signature)
        except Exception as e:
            logger.error(f"Error verificando firma de Facebook: {e}")
            return False
    
    def _verify_instagram_signature(
        self,
        payload: Dict[str, Any],
        signature: str,
        secret: str
    ) -> bool:
        """Verificar firma de Instagram"""
        # Similar a Facebook
        return self._verify_facebook_signature(payload, signature, secret)
    
    def _extract_event_type(
        self,
        platform: str,
        payload: Dict[str, Any]
    ) -> Optional[str]:
        """
        Extraer tipo de evento del payload
        
        Args:
            platform: Plataforma
            payload: Datos del webhook
            
        Returns:
            Tipo de evento o None
        """
        if platform == "facebook":
            entry = payload.get("entry", [{}])[0]
            return entry.get("changes", [{}])[0].get("field")
        elif platform == "instagram":
            entry = payload.get("entry", [{}])[0]
            return entry.get("changes", [{}])[0].get("field")
        elif platform == "twitter":
            return payload.get("event")
        
        return payload.get("type") or payload.get("event_type")
    
    def set_verification_token(self, platform: str, token: str):
        """
        Configurar token de verificación
        
        Args:
            platform: Plataforma
            token: Token de verificación
        """
        self.verification_tokens[platform] = token
        logger.info(f"Token de verificación configurado para {platform}")
    
    def get_webhook_history(
        self,
        platform: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Obtener historial de webhooks procesados
        
        Args:
            platform: Filtrar por plataforma
            limit: Límite de resultados
            
        Returns:
            Lista de webhooks procesados
        """
        if not hasattr(self, 'webhook_history'):
            self.webhook_history = []
        
        history = self.webhook_history
        
        if platform:
            history = [w for w in history if w.get("platform") == platform]
        
        return history[-limit:]
    
    def _record_webhook(self, platform: str, event_type: str, payload: Dict[str, Any]):
        """Registrar webhook en historial"""
        if not hasattr(self, 'webhook_history'):
            self.webhook_history = []
        
        from datetime import datetime
        
        self.webhook_history.append({
            "platform": platform,
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "payload_size": len(str(payload))
        })
        
        if len(self.webhook_history) > 1000:
            self.webhook_history = self.webhook_history[-1000:]
    
    def handle_webhook(
        self,
        platform: str,
        payload: Dict[str, Any],
        signature: Optional[str] = None
    ) -> bool:
        """
        Manejar un webhook
        
        Args:
            platform: Plataforma
            payload: Datos del webhook
            signature: Firma para verificación (opcional)
            
        Returns:
            True si se manejó exitosamente
        """
        # Verificar firma si está disponible
        if signature and not self.verify_signature(platform, payload, signature):
            logger.warning(f"Firma inválida para webhook de {platform}")
            return False
        
        # Determinar tipo de evento
        event_type = self._extract_event_type(platform, payload)
        
        if not event_type:
            logger.warning(f"No se pudo determinar tipo de evento para {platform}")
            return False
        
        # Registrar en historial
        self._record_webhook(platform, event_type, payload)
        
        # Ejecutar handlers
        key = f"{platform}:{event_type}"
        handlers = self.handlers.get(key, [])
        
        if not handlers:
            logger.debug(f"No hay handlers para {key}")
            return False
        
        for handler in handlers:
            try:
                handler(payload)
            except Exception as e:
                logger.error(f"Error en handler de webhook: {e}")
        
        logger.info(f"Webhook manejado: {key}")
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de webhooks
        
        Returns:
            Dict con estadísticas
        """
        if not hasattr(self, 'webhook_history'):
            return {
                "total_webhooks": 0,
                "by_platform": {},
                "by_event_type": {},
                "handlers_registered": sum(len(h) for h in self.handlers.values())
            }
        
        by_platform = defaultdict(int)
        by_event_type = defaultdict(int)
        
        for webhook in self.webhook_history:
            by_platform[webhook.get("platform", "unknown")] += 1
            by_event_type[webhook.get("event_type", "unknown")] += 1
        
        return {
            "total_webhooks": len(self.webhook_history),
            "by_platform": dict(by_platform),
            "by_event_type": dict(by_event_type),
            "handlers_registered": sum(len(h) for h in self.handlers.values()),
            "platforms_with_tokens": len(self.verification_tokens)
        }
    
    def verify_webhook_challenge(
        self,
        platform: str,
        challenge: str,
        verify_token: str
    ) -> Optional[str]:
        """
        Verificar challenge de webhook (para verificación inicial)
        
        Args:
            platform: Plataforma
            challenge: Challenge recibido
            verify_token: Token de verificación esperado
            
        Returns:
            Challenge si es válido, None si no
        """
        expected_token = self.verification_tokens.get(platform)
        
        if expected_token and verify_token == expected_token:
            logger.info(f"Challenge verificado para {platform}")
            return challenge
        
        logger.warning(f"Challenge inválido para {platform}")
        return None

