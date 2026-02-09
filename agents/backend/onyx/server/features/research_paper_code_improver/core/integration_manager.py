"""
Integration Manager - Gestión de integraciones con servicios externos
=======================================================================
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import httpx
import asyncio

logger = logging.getLogger(__name__)


class IntegrationManager:
    """
    Gestiona integraciones con servicios externos (Slack, Discord, Email, etc.)
    """
    
    def __init__(self):
        """Inicializar gestor de integraciones"""
        self.integrations: Dict[str, Dict[str, Any]] = {}
        self.client = httpx.AsyncClient(timeout=10.0)
    
    def register_integration(
        self,
        service: str,
        config: Dict[str, Any]
    ) -> bool:
        """
        Registra una integración.
        
        Args:
            service: Nombre del servicio (slack, discord, email, etc.)
            config: Configuración de la integración
            
        Returns:
            True si se registró exitosamente
        """
        try:
            self.integrations[service] = {
                "service": service,
                "config": config,
                "enabled": config.get("enabled", True),
                "created_at": datetime.now().isoformat()
            }
            
            logger.info(f"Integración registrada: {service}")
            return True
            
        except Exception as e:
            logger.error(f"Error registrando integración: {e}")
            return False
    
    async def send_notification(
        self,
        service: str,
        message: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Envía notificación a un servicio.
        
        Args:
            service: Nombre del servicio
            message: Mensaje a enviar
            title: Título (opcional)
            metadata: Metadata adicional (opcional)
            
        Returns:
            True si se envió exitosamente
        """
        if service not in self.integrations:
            logger.warning(f"Integración no registrada: {service}")
            return False
        
        integration = self.integrations[service]
        
        if not integration.get("enabled", True):
            logger.debug(f"Integración deshabilitada: {service}")
            return False
        
        try:
            if service == "slack":
                return await self._send_slack(message, title, integration["config"], metadata)
            elif service == "discord":
                return await self._send_discord(message, title, integration["config"], metadata)
            elif service == "email":
                return await self._send_email(message, title, integration["config"], metadata)
            else:
                logger.warning(f"Servicio no soportado: {service}")
                return False
                
        except Exception as e:
            logger.error(f"Error enviando notificación a {service}: {e}")
            return False
    
    async def _send_slack(
        self,
        message: str,
        title: Optional[str],
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]]
    ) -> bool:
        """Envía mensaje a Slack"""
        webhook_url = config.get("webhook_url")
        if not webhook_url:
            return False
        
        payload = {
            "text": title or "Research Paper Code Improver",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": message
                    }
                }
            ]
        }
        
        if metadata:
            payload["blocks"].append({
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*{k}*: {v}"}
                    for k, v in metadata.items()
                ]
            })
        
        response = await self.client.post(webhook_url, json=payload)
        response.raise_for_status()
        
        return True
    
    async def _send_discord(
        self,
        message: str,
        title: Optional[str],
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]]
    ) -> bool:
        """Envía mensaje a Discord"""
        webhook_url = config.get("webhook_url")
        if not webhook_url:
            return False
        
        embed = {
            "title": title or "Research Paper Code Improver",
            "description": message,
            "color": 0x667eea,
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata:
            embed["fields"] = [
                {"name": k, "value": str(v), "inline": True}
                for k, v in metadata.items()
            ]
        
        payload = {"embeds": [embed]}
        
        response = await self.client.post(webhook_url, json=payload)
        response.raise_for_status()
        
        return True
    
    async def _send_email(
        self,
        message: str,
        title: Optional[str],
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]]
    ) -> bool:
        """Envía email (placeholder - requiere servicio de email)"""
        # En producción, esto usaría un servicio de email real
        logger.info(f"Email enviado: {title} - {message}")
        return True
    
    async def send_to_all(
        self,
        message: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, bool]:
        """
        Envía notificación a todos los servicios habilitados.
        
        Args:
            message: Mensaje
            title: Título (opcional)
            metadata: Metadata (opcional)
            
        Returns:
            Resultados por servicio
        """
        results = {}
        
        for service in self.integrations.keys():
            if self.integrations[service].get("enabled", True):
                success = await self.send_notification(service, message, title, metadata)
                results[service] = success
        
        return results
    
    def list_integrations(self) -> List[Dict[str, Any]]:
        """Lista integraciones registradas"""
        return [
            {
                "service": integration["service"],
                "enabled": integration.get("enabled", True),
                "created_at": integration.get("created_at", "")
            }
            for integration in self.integrations.values()
        ]
    
    async def close(self):
        """Cierra el cliente HTTP"""
        await self.client.aclose()

