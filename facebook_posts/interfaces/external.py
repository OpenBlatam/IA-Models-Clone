from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from datetime import datetime
from ..models.facebook_models import FacebookPostEntity
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
🎯 External Service Interfaces
==============================

Interfaces para servicios externos e integraciones.
"""




class FacebookAPIInterface(ABC):
    """Interface para Facebook API."""
    
    @abstractmethod
    async def publish_post(self, post: FacebookPostEntity, page_id: str) -> Dict[str, Any]:
        """Publicar post en Facebook."""
        pass
    
    @abstractmethod
    async def get_post_metrics(self, post_id: str) -> Dict[str, Any]:
        """Obtener métricas de post publicado."""
        pass
    
    @abstractmethod
    async def schedule_post(self, post: FacebookPostEntity, publish_time: datetime) -> Dict[str, Any]:
        """Programar publicación de post."""
        pass


class OnySIntegrationInterface(ABC):
    """Interface para integración con Onyx."""
    
    @abstractmethod
    async def get_workspace_context(self, workspace_id: str) -> Dict[str, Any]:
        """Obtener contexto del workspace."""
        pass
    
    @abstractmethod
    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Obtener preferencias del usuario."""
        pass
    
    @abstractmethod
    async def log_activity(self, user_id: str, activity: str, metadata: Dict[str, Any]) -> bool:
        """Registrar actividad del usuario."""
        pass
    
    @abstractmethod
    async def send_notification(self, user_id: str, message: str, type: str = "info") -> bool:
        """Enviar notificación al usuario."""
        pass


class AIModelInterface(ABC):
    """Interface para modelos de AI."""
    
    @abstractmethod
    async def generate_text(self, prompt: str, max_tokens: int = 500) -> str:
        """Generar texto con modelo AI."""
        pass
    
    @abstractmethod
    async def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analizar sentimiento del texto."""
        pass
    
    @abstractmethod
    async def extract_keywords(self, text: str) -> List[str]:
        """Extraer keywords del texto."""
        pass
    
    @abstractmethod
    async def calculate_readability(self, text: str) -> float:
        """Calcular score de legibilidad."""
        pass 