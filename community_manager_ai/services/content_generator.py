"""
Content Generator - Generador de Contenido
===========================================

Sistema para generar contenido automáticamente usando IA.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class ContentGenerator:
    """Generador de contenido para redes sociales"""
    
    def __init__(self, ai_model: Optional[Any] = None):
        """
        Inicializar el generador de contenido
        
        Args:
            ai_model: Modelo de IA para generar contenido (opcional)
        """
        self.ai_model = ai_model
        self._ai_generator = None
        self._init_ai_generator()
        logger.info("Content Generator inicializado")
    
    def _init_ai_generator(self):
        """Inicializar generador de IA si está disponible"""
        try:
            from .ai_content_generator import AIContentGenerator
            from ..config import get_settings
            
            settings = get_settings()
            
            if not getattr(settings, 'ai_enabled', True):
                logger.debug("AI deshabilitado en configuración")
                return
            
            ai_provider = getattr(settings, 'ai_provider', 'openai')
            
            if ai_provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY") or getattr(settings, 'openai_api_key', None)
            elif ai_provider == "anthropic":
                api_key = os.getenv("ANTHROPIC_API_KEY") or getattr(settings, 'anthropic_api_key', None)
            else:
                api_key = None
            
            if api_key:
                self._ai_generator = AIContentGenerator(
                    ai_provider=ai_provider,
                    api_key=api_key
                )
                logger.info(f"AI Content Generator inicializado con {ai_provider}")
            else:
                logger.debug(f"{ai_provider} API key no configurada, usando modo fallback")
        except Exception as e:
            logger.debug(f"No se pudo inicializar AI generator: {e}")
    
    def generate_post(
        self,
        topic: str,
        platform: str,
        tone: Optional[str] = "professional",
        length: Optional[str] = "medium",
        context: Optional[str] = None
    ) -> str:
        """
        Generar un post para una plataforma
        
        Args:
            topic: Tema del post
            platform: Plataforma objetivo
            tone: Tono del contenido (professional, casual, funny, etc.)
            length: Longitud (short, medium, long)
            context: Contexto adicional para la generación
            
        Returns:
            Contenido generado
        """
        if self._ai_generator:
            try:
                return self._ai_generator.generate_post(
                    topic=topic,
                    platform=platform,
                    tone=tone,
                    length=length,
                    context=context
                )
            except Exception as e:
                logger.warning(f"Error usando IA generator, usando fallback: {e}")
        
        templates = {
            "short": f"📢 {topic} - ¡No te lo pierdas!",
            "medium": f"🎯 Hoy queremos hablar sobre {topic}. Es un tema importante que afecta a muchos. ¿Qué opinas?",
            "long": f"🌟 Hablemos sobre {topic}\n\nEste es un tema que nos parece muy relevante en la actualidad. Queremos compartir nuestra perspectiva y conocer la tuya.\n\n¿Has tenido alguna experiencia relacionada? Comparte en los comentarios."
        }
        
        content = templates.get(length, templates["medium"])
        return self.optimize_for_platform(content, platform)
    
    def generate_caption(
        self,
        image_description: str,
        platform: str,
        hashtags: Optional[List[str]] = None
    ) -> str:
        """
        Generar caption para una imagen
        
        Args:
            image_description: Descripción de la imagen
            platform: Plataforma objetivo
            hashtags: Lista de hashtags
            
        Returns:
            Caption generado
        """
        if self._ai_generator:
            try:
                return self._ai_generator.generate_caption(
                    image_description=image_description,
                    platform=platform,
                    hashtags=hashtags
                )
            except Exception as e:
                logger.warning(f"Error usando IA generator, usando fallback: {e}")
        
        caption = f"✨ {image_description}\n\n"
        
        if hashtags:
            hashtag_str = " ".join([f"#{tag}" if not tag.startswith("#") else tag for tag in hashtags])
            caption += hashtag_str
        
        return self.optimize_for_platform(caption, platform)
    
    def generate_hashtags(
        self,
        content: str,
        platform: str,
        count: int = 10
    ) -> List[str]:
        """
        Generar hashtags relevantes
        
        Args:
            content: Contenido del post
            platform: Plataforma objetivo
            count: Número de hashtags a generar
            
        Returns:
            Lista de hashtags
        """
        if self._ai_generator:
            try:
                hashtags = self._ai_generator.generate_hashtags(
                    content=content,
                    platform=platform,
                    count=count
                )
                return [f"#{tag}" if not tag.startswith("#") else tag for tag in hashtags]
            except Exception as e:
                logger.warning(f"Error usando IA generator, usando fallback: {e}")
        
        from ..utils.content_optimizer import ContentOptimizer
        try:
            keywords = ContentOptimizer.extract_keywords(content, count)
            return [f"#{tag}" if not tag.startswith("#") else tag for tag in keywords]
        except Exception:
            words = content.lower().split()
            hashtags = [word.strip('.,!?;:') for word in words if len(word.strip('.,!?;:')) > 4][:count]
            return [f"#{tag}" for tag in hashtags]
    
    def optimize_for_platform(
        self,
        content: str,
        platform: str
    ) -> str:
        """
        Optimizar contenido para una plataforma específica
        
        Args:
            content: Contenido original
            platform: Plataforma objetivo
            
        Returns:
            Contenido optimizado
        """
        # Reglas específicas por plataforma
        optimizations = {
            "twitter": self._optimize_twitter,
            "instagram": self._optimize_instagram,
            "facebook": self._optimize_facebook,
            "linkedin": self._optimize_linkedin,
            "tiktok": self._optimize_tiktok
        }
        
        optimizer = optimizations.get(platform.lower(), lambda x: x)
        return optimizer(content)
    
    def _optimize_twitter(self, content: str) -> str:
        """Optimizar para Twitter (280 caracteres)"""
        if len(content) > 280:
            return content[:277] + "..."
        return content
    
    def _optimize_instagram(self, content: str) -> str:
        """Optimizar para Instagram"""
        from ..utils.content_optimizer import ContentOptimizer
        optimized = ContentOptimizer.optimize_length(content, "instagram")
        return optimized
    
    def _optimize_facebook(self, content: str) -> str:
        """Optimizar para Facebook"""
        from ..utils.content_optimizer import ContentOptimizer
        optimized = ContentOptimizer.optimize_length(content, "facebook")
        return optimized
    
    def _optimize_linkedin(self, content: str) -> str:
        """Optimizar para LinkedIn (más profesional)"""
        from ..utils.content_optimizer import ContentOptimizer
        optimized = ContentOptimizer.optimize_length(content, "linkedin")
        return optimized
    
    def _optimize_tiktok(self, content: str) -> str:
        """Optimizar para TikTok"""
        from ..utils.content_optimizer import ContentOptimizer
        optimized = ContentOptimizer.optimize_length(content, "tiktok")
        return optimized



