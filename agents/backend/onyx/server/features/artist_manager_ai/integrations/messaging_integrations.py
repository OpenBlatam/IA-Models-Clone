"""
Messaging Integrations
======================

Integraciones con servicios de mensajería (WhatsApp, Telegram).
"""

import logging
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class MessagingIntegration(ABC):
    """Clase base para integraciones de mensajería."""
    
    def __init__(self, credentials: Dict[str, Any]):
        """
        Inicializar integración.
        
        Args:
            credentials: Credenciales de la integración
        """
        self.credentials = credentials
        self._logger = logger
    
    @abstractmethod
    async def send_message(self, recipient: str, message: str) -> bool:
        """
        Enviar mensaje.
        
        Args:
            recipient: Destinatario
            message: Mensaje
        
        Returns:
            True si se envió
        """
        pass
    
    @abstractmethod
    async def send_notification(self, recipient: str, title: str, message: str) -> bool:
        """
        Enviar notificación.
        
        Args:
            recipient: Destinatario
            title: Título
            message: Mensaje
        
        Returns:
            True si se envió
        """
        pass


class WhatsAppIntegration(MessagingIntegration):
    """Integración con WhatsApp (usando API de terceros)."""
    
    def __init__(self, credentials: Dict[str, Any]):
        """
        Inicializar integración con WhatsApp.
        
        Args:
            credentials: Debe contener 'api_key' y 'api_url'
        """
        super().__init__(credentials)
        self.api_key = credentials.get("api_key")
        self.api_url = credentials.get("api_url", "https://api.whatsapp.com/v1")
        self.phone_number = credentials.get("phone_number")
    
    async def send_message(self, recipient: str, message: str) -> bool:
        """Enviar mensaje por WhatsApp."""
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_url}/messages",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "to": recipient,
                        "from": self.phone_number,
                        "body": message
                    }
                )
                response.raise_for_status()
                self._logger.info(f"Sent WhatsApp message to {recipient}")
                return True
        
        except Exception as e:
            self._logger.error(f"Error sending WhatsApp message: {str(e)}")
            return False
    
    async def send_notification(self, recipient: str, title: str, message: str) -> bool:
        """Enviar notificación por WhatsApp."""
        full_message = f"*{title}*\n\n{message}"
        return await self.send_message(recipient, full_message)


class TelegramIntegration(MessagingIntegration):
    """Integración con Telegram."""
    
    def __init__(self, credentials: Dict[str, Any]):
        """
        Inicializar integración con Telegram.
        
        Args:
            credentials: Debe contener 'bot_token' y opcionalmente 'chat_id'
        """
        super().__init__(credentials)
        self.bot_token = credentials.get("bot_token")
        self.chat_id = credentials.get("chat_id")
        self.api_base = f"https://api.telegram.org/bot{self.bot_token}"
    
    async def send_message(self, recipient: str, message: str) -> bool:
        """Enviar mensaje por Telegram."""
        try:
            import httpx
            
            chat_id = recipient or self.chat_id
            if not chat_id:
                raise ValueError("No chat_id provided")
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base}/sendMessage",
                    json={
                        "chat_id": chat_id,
                        "text": message,
                        "parse_mode": "Markdown"
                    }
                )
                response.raise_for_status()
                self._logger.info(f"Sent Telegram message to {chat_id}")
                return True
        
        except Exception as e:
            self._logger.error(f"Error sending Telegram message: {str(e)}")
            return False
    
    async def send_notification(self, recipient: str, title: str, message: str) -> bool:
        """Enviar notificación por Telegram."""
        full_message = f"*{title}*\n\n{message}"
        return await self.send_message(recipient, full_message)




